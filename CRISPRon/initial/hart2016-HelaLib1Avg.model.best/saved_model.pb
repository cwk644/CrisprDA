��
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
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
�i
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�i
value�iB�i B�h
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
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
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
h

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
R
.regularization_losses
/trainable_variables
0	variables
1	keras_api
R
2regularization_losses
3trainable_variables
4	variables
5	keras_api
R
6regularization_losses
7trainable_variables
8	variables
9	keras_api
R
:regularization_losses
;trainable_variables
<	variables
=	keras_api
R
>regularization_losses
?trainable_variables
@	variables
A	keras_api
R
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
R
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
R
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
R
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
R
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
h

Vkernel
Wbias
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
R
\regularization_losses
]trainable_variables
^	variables
_	keras_api
h

`kernel
abias
bregularization_losses
ctrainable_variables
d	variables
e	keras_api
R
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
h

jkernel
kbias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
R
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
h

tkernel
ubias
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
�
ziter

{beta_1

|beta_2
	}decay
~learning_ratem�m�"m�#m�(m�)m�Vm�Wm�`m�am�jm�km�tm�um�v�v�"v�#v�(v�)v�Vv�Wv�`v�av�jv�kv�tv�uv�
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
regularization_losses
�non_trainable_variables
trainable_variables
�metrics
 �layer_regularization_losses
	variables
�layers
 
YW
VARIABLE_VALUEconv_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
�layer_metrics
regularization_losses
�non_trainable_variables
trainable_variables
�metrics
 �layer_regularization_losses
 	variables
�layers
YW
VARIABLE_VALUEconv_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
�
�layer_metrics
$regularization_losses
�non_trainable_variables
%trainable_variables
�metrics
 �layer_regularization_losses
&	variables
�layers
YW
VARIABLE_VALUEconv_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
�
�layer_metrics
*regularization_losses
�non_trainable_variables
+trainable_variables
�metrics
 �layer_regularization_losses
,	variables
�layers
 
 
 
�
�layer_metrics
.regularization_losses
�non_trainable_variables
/trainable_variables
�metrics
 �layer_regularization_losses
0	variables
�layers
 
 
 
�
�layer_metrics
2regularization_losses
�non_trainable_variables
3trainable_variables
�metrics
 �layer_regularization_losses
4	variables
�layers
 
 
 
�
�layer_metrics
6regularization_losses
�non_trainable_variables
7trainable_variables
�metrics
 �layer_regularization_losses
8	variables
�layers
 
 
 
�
�layer_metrics
:regularization_losses
�non_trainable_variables
;trainable_variables
�metrics
 �layer_regularization_losses
<	variables
�layers
 
 
 
�
�layer_metrics
>regularization_losses
�non_trainable_variables
?trainable_variables
�metrics
 �layer_regularization_losses
@	variables
�layers
 
 
 
�
�layer_metrics
Bregularization_losses
�non_trainable_variables
Ctrainable_variables
�metrics
 �layer_regularization_losses
D	variables
�layers
 
 
 
�
�layer_metrics
Fregularization_losses
�non_trainable_variables
Gtrainable_variables
�metrics
 �layer_regularization_losses
H	variables
�layers
 
 
 
�
�layer_metrics
Jregularization_losses
�non_trainable_variables
Ktrainable_variables
�metrics
 �layer_regularization_losses
L	variables
�layers
 
 
 
�
�layer_metrics
Nregularization_losses
�non_trainable_variables
Otrainable_variables
�metrics
 �layer_regularization_losses
P	variables
�layers
 
 
 
�
�layer_metrics
Rregularization_losses
�non_trainable_variables
Strainable_variables
�metrics
 �layer_regularization_losses
T	variables
�layers
ZX
VARIABLE_VALUEdense_0/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_0/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

V0
W1

V0
W1
�
�layer_metrics
Xregularization_losses
�non_trainable_variables
Ytrainable_variables
�metrics
 �layer_regularization_losses
Z	variables
�layers
 
 
 
�
�layer_metrics
\regularization_losses
�non_trainable_variables
]trainable_variables
�metrics
 �layer_regularization_losses
^	variables
�layers
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

`0
a1

`0
a1
�
�layer_metrics
bregularization_losses
�non_trainable_variables
ctrainable_variables
�metrics
 �layer_regularization_losses
d	variables
�layers
 
 
 
�
�layer_metrics
fregularization_losses
�non_trainable_variables
gtrainable_variables
�metrics
 �layer_regularization_losses
h	variables
�layers
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

j0
k1

j0
k1
�
�layer_metrics
lregularization_losses
�non_trainable_variables
mtrainable_variables
�metrics
 �layer_regularization_losses
n	variables
�layers
 
 
 
�
�layer_metrics
pregularization_losses
�non_trainable_variables
qtrainable_variables
�metrics
 �layer_regularization_losses
r	variables
�layers
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

t0
u1

t0
u1
�
�layer_metrics
vregularization_losses
�non_trainable_variables
wtrainable_variables
�metrics
 �layer_regularization_losses
x	variables
�layers
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

�0
�1
�2
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
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
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
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 
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
GPU2 *0J 8� *-
f(R&
$__inference_signature_wrapper_845303
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv_3/kernel/Read/ReadVariableOpconv_3/bias/Read/ReadVariableOp!conv_5/kernel/Read/ReadVariableOpconv_5/bias/Read/ReadVariableOp!conv_7/kernel/Read/ReadVariableOpconv_7/bias/Read/ReadVariableOp"dense_0/kernel/Read/ReadVariableOp dense_0/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp(Adam/conv_3/kernel/m/Read/ReadVariableOp&Adam/conv_3/bias/m/Read/ReadVariableOp(Adam/conv_5/kernel/m/Read/ReadVariableOp&Adam/conv_5/bias/m/Read/ReadVariableOp(Adam/conv_7/kernel/m/Read/ReadVariableOp&Adam/conv_7/bias/m/Read/ReadVariableOp)Adam/dense_0/kernel/m/Read/ReadVariableOp'Adam/dense_0/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp(Adam/conv_3/kernel/v/Read/ReadVariableOp&Adam/conv_3/bias/v/Read/ReadVariableOp(Adam/conv_5/kernel/v/Read/ReadVariableOp&Adam/conv_5/bias/v/Read/ReadVariableOp(Adam/conv_7/kernel/v/Read/ReadVariableOp&Adam/conv_7/bias/v/Read/ReadVariableOp)Adam/dense_0/kernel/v/Read/ReadVariableOp'Adam/dense_0/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*B
Tin;
927	*
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
GPU2 *0J 8� *(
f#R!
__inference__traced_save_846143
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_3/kernelconv_3/biasconv_5/kernelconv_5/biasconv_7/kernelconv_7/biasdense_0/kerneldense_0/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/conv_3/kernel/mAdam/conv_3/bias/mAdam/conv_5/kernel/mAdam/conv_5/bias/mAdam/conv_7/kernel/mAdam/conv_7/bias/mAdam/dense_0/kernel/mAdam/dense_0/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv_3/kernel/vAdam/conv_3/bias/vAdam/conv_5/kernel/vAdam/conv_5/bias/vAdam/conv_7/kernel/vAdam/conv_7/bias/vAdam/dense_0/kernel/vAdam/dense_0/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/output/kernel/vAdam/output/bias/v*A
Tin:
826*
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
GPU2 *0J 8� *+
f&R$
"__inference__traced_restore_846312��
�
C
'__inference_pool_7_layer_call_fn_844522

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
GPU2 *0J 8� *K
fFRD
B__inference_pool_7_layer_call_and_return_conditional_losses_8445162
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
��
�
"__inference__traced_restore_846312
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
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: %
assignvariableop_23_total_2: %
assignvariableop_24_count_2: >
(assignvariableop_25_adam_conv_3_kernel_m:d4
&assignvariableop_26_adam_conv_3_bias_m:d>
(assignvariableop_27_adam_conv_5_kernel_m:F4
&assignvariableop_28_adam_conv_5_bias_m:F>
(assignvariableop_29_adam_conv_7_kernel_m:(4
&assignvariableop_30_adam_conv_7_bias_m:(<
)assignvariableop_31_adam_dense_0_kernel_m:	�P5
'assignvariableop_32_adam_dense_0_bias_m:P;
)assignvariableop_33_adam_dense_1_kernel_m:PP5
'assignvariableop_34_adam_dense_1_bias_m:P;
)assignvariableop_35_adam_dense_2_kernel_m:P<5
'assignvariableop_36_adam_dense_2_bias_m:<:
(assignvariableop_37_adam_output_kernel_m:<4
&assignvariableop_38_adam_output_bias_m:>
(assignvariableop_39_adam_conv_3_kernel_v:d4
&assignvariableop_40_adam_conv_3_bias_v:d>
(assignvariableop_41_adam_conv_5_kernel_v:F4
&assignvariableop_42_adam_conv_5_bias_v:F>
(assignvariableop_43_adam_conv_7_kernel_v:(4
&assignvariableop_44_adam_conv_7_bias_v:(<
)assignvariableop_45_adam_dense_0_kernel_v:	�P5
'assignvariableop_46_adam_dense_0_bias_v:P;
)assignvariableop_47_adam_dense_1_kernel_v:PP5
'assignvariableop_48_adam_dense_1_bias_v:P;
)assignvariableop_49_adam_dense_2_kernel_v:P<5
'assignvariableop_50_adam_dense_2_bias_v:<:
(assignvariableop_51_adam_output_kernel_v:<4
&assignvariableop_52_adam_output_bias_v:
identity_54��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*�
value�B�6B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
826	2
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
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_2Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_conv_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_conv_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_conv_5_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_conv_5_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_conv_7_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_conv_7_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_0_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_0_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_2_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_2_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_output_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_output_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_conv_3_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_conv_3_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_conv_5_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_conv_5_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_conv_7_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_conv_7_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_0_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_0_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_1_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_1_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_2_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_2_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_output_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp&assignvariableop_52_adam_output_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_529
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�	
Identity_53Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_53�	
Identity_54IdentityIdentity_53:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_54"#
identity_54Identity_54:output:0*
_input_shapesn
l: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
`
'__inference_drop_3_layer_call_fn_845682

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
GPU2 *0J 8� *K
fFRD
B__inference_drop_3_layer_call_and_return_conditional_losses_8449282
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
�
a
B__inference_drop_5_layer_call_and_return_conditional_losses_844951

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
C__inference_drop_d1_layer_call_and_return_conditional_losses_845895

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
B__inference_conv_3_layer_call_and_return_conditional_losses_845622

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
�
a
C__inference_drop_d1_layer_call_and_return_conditional_losses_845883

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
�
a
(__inference_drop_d2_layer_call_fn_845925

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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d2_layer_call_and_return_conditional_losses_8448032
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
�
a
(__inference_drop_d1_layer_call_fn_845878

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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d1_layer_call_and_return_conditional_losses_8448362
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
�
^
B__inference_pool_3_layer_call_and_return_conditional_losses_844486

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
C
'__inference_drop_3_layer_call_fn_845677

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
GPU2 *0J 8� *K
fFRD
B__inference_drop_3_layer_call_and_return_conditional_losses_8446142
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
�
�
(__inference_dense_2_layer_call_fn_845904

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
GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_8447122
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
�
`
'__inference_drop_5_layer_call_fn_845709

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
GPU2 *0J 8� *K
fFRD
B__inference_drop_5_layer_call_and_return_conditional_losses_8449512
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
�
�
B__inference_conv_7_layer_call_and_return_conditional_losses_844545

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
�
�
B__inference_conv_3_layer_call_and_return_conditional_losses_844589

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
�
h
.__inference_concatenate_4_layer_call_fn_845793
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
GPU2 *0J 8� *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_8446512
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
(__inference_model_4_layer_call_fn_845336

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
GPU2 *0J 8� *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_8447422
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
�
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_845775

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
�
B__inference_output_layer_call_and_return_conditional_losses_845961

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
�
B__inference_output_layer_call_and_return_conditional_losses_844735

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
�
`
B__inference_drop_5_layer_call_and_return_conditional_losses_844607

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
�
`
B__inference_drop_3_layer_call_and_return_conditional_losses_845687

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
a
B__inference_drop_3_layer_call_and_return_conditional_losses_844928

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
b
C__inference_drop_d0_layer_call_and_return_conditional_losses_845848

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
�
a
C__inference_drop_d0_layer_call_and_return_conditional_losses_844675

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
C__inference_dense_2_layer_call_and_return_conditional_losses_844712

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
�
`
'__inference_drop_7_layer_call_fn_845736

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
GPU2 *0J 8� *K
fFRD
B__inference_drop_7_layer_call_and_return_conditional_losses_8449742
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
�
�
(__inference_model_4_layer_call_fn_845369

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
GPU2 *0J 8� *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_8450942
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
�
�
'__inference_conv_3_layer_call_fn_845606

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
GPU2 *0J 8� *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_8445892
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
`
B__inference_drop_7_layer_call_and_return_conditional_losses_844600

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
�
�
$__inference_signature_wrapper_845303
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
GPU2 *0J 8� **
f%R#
!__inference__wrapped_model_8444772
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
�
F
*__inference_flatten_5_layer_call_fn_845769

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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_8446332
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
a
B__inference_drop_7_layer_call_and_return_conditional_losses_844974

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
�
a
B__inference_drop_3_layer_call_and_return_conditional_losses_845699

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
b
C__inference_drop_d0_layer_call_and_return_conditional_losses_844869

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
�L
�
C__inference_model_4_layer_call_and_return_conditional_losses_844742

inputs#
conv_7_844546:(
conv_7_844548:(#
conv_5_844568:F
conv_5_844570:F#
conv_3_844590:d
conv_3_844592:d!
dense_0_844665:	�P
dense_0_844667:P 
dense_1_844689:PP
dense_1_844691:P 
dense_2_844713:P<
dense_2_844715:<
output_844736:<
output_844738:
identity��conv_3/StatefulPartitionedCall�conv_5/StatefulPartitionedCall�conv_7/StatefulPartitionedCall�dense_0/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�output/StatefulPartitionedCall�
conv_7/StatefulPartitionedCallStatefulPartitionedCallinputsconv_7_844546conv_7_844548*
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
GPU2 *0J 8� *K
fFRD
B__inference_conv_7_layer_call_and_return_conditional_losses_8445452 
conv_7/StatefulPartitionedCall�
conv_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv_5_844568conv_5_844570*
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
GPU2 *0J 8� *K
fFRD
B__inference_conv_5_layer_call_and_return_conditional_losses_8445672 
conv_5/StatefulPartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv_3_844590conv_3_844592*
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
GPU2 *0J 8� *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_8445892 
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
GPU2 *0J 8� *K
fFRD
B__inference_drop_7_layer_call_and_return_conditional_losses_8446002
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
GPU2 *0J 8� *K
fFRD
B__inference_drop_5_layer_call_and_return_conditional_losses_8446072
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
GPU2 *0J 8� *K
fFRD
B__inference_drop_3_layer_call_and_return_conditional_losses_8446142
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
GPU2 *0J 8� *K
fFRD
B__inference_pool_7_layer_call_and_return_conditional_losses_8445162
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
GPU2 *0J 8� *K
fFRD
B__inference_pool_5_layer_call_and_return_conditional_losses_8445012
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
GPU2 *0J 8� *K
fFRD
B__inference_pool_3_layer_call_and_return_conditional_losses_8444862
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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_8446252
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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_8446332
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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_8446412
flatten_7/PartitionedCall�
concatenate_4/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0*
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
GPU2 *0J 8� *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_8446512
concatenate_4/PartitionedCall�
dense_0/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_0_844665dense_0_844667*
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
GPU2 *0J 8� *L
fGRE
C__inference_dense_0_layer_call_and_return_conditional_losses_8446642!
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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d0_layer_call_and_return_conditional_losses_8446752
drop_d0/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall drop_d0/PartitionedCall:output:0dense_1_844689dense_1_844691*
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
GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_8446882!
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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d1_layer_call_and_return_conditional_losses_8446992
drop_d1/PartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall drop_d1/PartitionedCall:output:0dense_2_844713dense_2_844715*
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
GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_8447122!
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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d2_layer_call_and_return_conditional_losses_8447232
drop_d2/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall drop_d2/PartitionedCall:output:0output_844736output_844738*
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
GPU2 *0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_8447352 
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
�
�
(__inference_model_4_layer_call_fn_844773
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
GPU2 *0J 8� *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_8447422
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
C__inference_dense_0_layer_call_and_return_conditional_losses_845821

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
'__inference_output_layer_call_fn_845951

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
GPU2 *0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_8447352
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
�U
�
C__inference_model_4_layer_call_and_return_conditional_losses_845262
input_onehot#
conv_7_845213:(
conv_7_845215:(#
conv_5_845218:F
conv_5_845220:F#
conv_3_845223:d
conv_3_845225:d!
dense_0_845238:	�P
dense_0_845240:P 
dense_1_845244:PP
dense_1_845246:P 
dense_2_845250:P<
dense_2_845252:<
output_845256:<
output_845258:
identity��conv_3/StatefulPartitionedCall�conv_5/StatefulPartitionedCall�conv_7/StatefulPartitionedCall�dense_0/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�drop_3/StatefulPartitionedCall�drop_5/StatefulPartitionedCall�drop_7/StatefulPartitionedCall�drop_d0/StatefulPartitionedCall�drop_d1/StatefulPartitionedCall�drop_d2/StatefulPartitionedCall�output/StatefulPartitionedCall�
conv_7/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv_7_845213conv_7_845215*
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
GPU2 *0J 8� *K
fFRD
B__inference_conv_7_layer_call_and_return_conditional_losses_8445452 
conv_7/StatefulPartitionedCall�
conv_5/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv_5_845218conv_5_845220*
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
GPU2 *0J 8� *K
fFRD
B__inference_conv_5_layer_call_and_return_conditional_losses_8445672 
conv_5/StatefulPartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv_3_845223conv_3_845225*
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
GPU2 *0J 8� *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_8445892 
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
GPU2 *0J 8� *K
fFRD
B__inference_drop_7_layer_call_and_return_conditional_losses_8449742 
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
GPU2 *0J 8� *K
fFRD
B__inference_drop_5_layer_call_and_return_conditional_losses_8449512 
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
GPU2 *0J 8� *K
fFRD
B__inference_drop_3_layer_call_and_return_conditional_losses_8449282 
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
GPU2 *0J 8� *K
fFRD
B__inference_pool_7_layer_call_and_return_conditional_losses_8445162
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
GPU2 *0J 8� *K
fFRD
B__inference_pool_5_layer_call_and_return_conditional_losses_8445012
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
GPU2 *0J 8� *K
fFRD
B__inference_pool_3_layer_call_and_return_conditional_losses_8444862
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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_8446252
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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_8446332
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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_8446412
flatten_7/PartitionedCall�
concatenate_4/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0*
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
GPU2 *0J 8� *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_8446512
concatenate_4/PartitionedCall�
dense_0/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_0_845238dense_0_845240*
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
GPU2 *0J 8� *L
fGRE
C__inference_dense_0_layer_call_and_return_conditional_losses_8446642!
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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d0_layer_call_and_return_conditional_losses_8448692!
drop_d0/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(drop_d0/StatefulPartitionedCall:output:0dense_1_845244dense_1_845246*
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
GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_8446882!
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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d1_layer_call_and_return_conditional_losses_8448362!
drop_d1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(drop_d1/StatefulPartitionedCall:output:0dense_2_845250dense_2_845252*
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
GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_8447122!
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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d2_layer_call_and_return_conditional_losses_8448032!
drop_d2/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall(drop_d2/StatefulPartitionedCall:output:0output_845256output_845258*
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
GPU2 *0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_8447352 
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
�
�
I__inference_concatenate_4_layer_call_and_return_conditional_losses_844651

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
�
C
'__inference_drop_7_layer_call_fn_845731

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
GPU2 *0J 8� *K
fFRD
B__inference_drop_7_layer_call_and_return_conditional_losses_8446002
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
�
�
(__inference_dense_0_layer_call_fn_845810

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
GPU2 *0J 8� *L
fGRE
C__inference_dense_0_layer_call_and_return_conditional_losses_8446642
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
�v
�
C__inference_model_4_layer_call_and_return_conditional_losses_845462

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
flatten_7/Reshapex
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_4/concat/axis�
concatenate_4/concatConcatV2flatten_3/Reshape:output:0flatten_5/Reshape:output:0flatten_7/Reshape:output:0"concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatenate_4/concat�
dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes
:	�P*
dtype02
dense_0/MatMul/ReadVariableOp�
dense_0/MatMulMatMulconcatenate_4/concat:output:0%dense_0/MatMul/ReadVariableOp:value:0*
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
�
a
B__inference_drop_7_layer_call_and_return_conditional_losses_845753

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
�
D
(__inference_drop_d1_layer_call_fn_845873

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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d1_layer_call_and_return_conditional_losses_8446992
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
�
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_844633

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
�
D
(__inference_drop_d0_layer_call_fn_845826

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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d0_layer_call_and_return_conditional_losses_8446752
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
F
*__inference_flatten_3_layer_call_fn_845758

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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_8446252
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
�
`
B__inference_drop_3_layer_call_and_return_conditional_losses_844614

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
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_844625

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
�
a
E__inference_flatten_7_layer_call_and_return_conditional_losses_844641

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
�
�
(__inference_dense_1_layer_call_fn_845857

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
GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_8446882
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
�
�
B__inference_conv_5_layer_call_and_return_conditional_losses_845647

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
a
C__inference_drop_d2_layer_call_and_return_conditional_losses_845930

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
�L
�
C__inference_model_4_layer_call_and_return_conditional_losses_845210
input_onehot#
conv_7_845161:(
conv_7_845163:(#
conv_5_845166:F
conv_5_845168:F#
conv_3_845171:d
conv_3_845173:d!
dense_0_845186:	�P
dense_0_845188:P 
dense_1_845192:PP
dense_1_845194:P 
dense_2_845198:P<
dense_2_845200:<
output_845204:<
output_845206:
identity��conv_3/StatefulPartitionedCall�conv_5/StatefulPartitionedCall�conv_7/StatefulPartitionedCall�dense_0/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�output/StatefulPartitionedCall�
conv_7/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv_7_845161conv_7_845163*
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
GPU2 *0J 8� *K
fFRD
B__inference_conv_7_layer_call_and_return_conditional_losses_8445452 
conv_7/StatefulPartitionedCall�
conv_5/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv_5_845166conv_5_845168*
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
GPU2 *0J 8� *K
fFRD
B__inference_conv_5_layer_call_and_return_conditional_losses_8445672 
conv_5/StatefulPartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv_3_845171conv_3_845173*
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
GPU2 *0J 8� *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_8445892 
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
GPU2 *0J 8� *K
fFRD
B__inference_drop_7_layer_call_and_return_conditional_losses_8446002
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
GPU2 *0J 8� *K
fFRD
B__inference_drop_5_layer_call_and_return_conditional_losses_8446072
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
GPU2 *0J 8� *K
fFRD
B__inference_drop_3_layer_call_and_return_conditional_losses_8446142
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
GPU2 *0J 8� *K
fFRD
B__inference_pool_7_layer_call_and_return_conditional_losses_8445162
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
GPU2 *0J 8� *K
fFRD
B__inference_pool_5_layer_call_and_return_conditional_losses_8445012
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
GPU2 *0J 8� *K
fFRD
B__inference_pool_3_layer_call_and_return_conditional_losses_8444862
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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_8446252
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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_8446332
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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_8446412
flatten_7/PartitionedCall�
concatenate_4/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0*
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
GPU2 *0J 8� *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_8446512
concatenate_4/PartitionedCall�
dense_0/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_0_845186dense_0_845188*
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
GPU2 *0J 8� *L
fGRE
C__inference_dense_0_layer_call_and_return_conditional_losses_8446642!
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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d0_layer_call_and_return_conditional_losses_8446752
drop_d0/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall drop_d0/PartitionedCall:output:0dense_1_845192dense_1_845194*
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
GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_8446882!
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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d1_layer_call_and_return_conditional_losses_8446992
drop_d1/PartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall drop_d1/PartitionedCall:output:0dense_2_845198dense_2_845200*
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
GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_8447122!
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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d2_layer_call_and_return_conditional_losses_8447232
drop_d2/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall drop_d2/PartitionedCall:output:0output_845204output_845206*
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
GPU2 *0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_8447352 
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
�
�
'__inference_conv_5_layer_call_fn_845631

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
GPU2 *0J 8� *K
fFRD
B__inference_conv_5_layer_call_and_return_conditional_losses_8445672
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
�
b
C__inference_drop_d1_layer_call_and_return_conditional_losses_844836

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
�
F
*__inference_flatten_7_layer_call_fn_845780

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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_8446412
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
�
�
I__inference_concatenate_4_layer_call_and_return_conditional_losses_845801
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
�
a
(__inference_drop_d0_layer_call_fn_845831

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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d0_layer_call_and_return_conditional_losses_8448692
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
�
a
C__inference_drop_d1_layer_call_and_return_conditional_losses_844699

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
C__inference_dense_2_layer_call_and_return_conditional_losses_845915

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
�
a
C__inference_drop_d2_layer_call_and_return_conditional_losses_844723

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
�h
�
__inference__traced_save_846143
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
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop3
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
:6*
dtype0*�
value�B�6B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop(savev2_conv_5_kernel_read_readvariableop&savev2_conv_5_bias_read_readvariableop(savev2_conv_7_kernel_read_readvariableop&savev2_conv_7_bias_read_readvariableop)savev2_dense_0_kernel_read_readvariableop'savev2_dense_0_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop/savev2_adam_conv_3_kernel_m_read_readvariableop-savev2_adam_conv_3_bias_m_read_readvariableop/savev2_adam_conv_5_kernel_m_read_readvariableop-savev2_adam_conv_5_bias_m_read_readvariableop/savev2_adam_conv_7_kernel_m_read_readvariableop-savev2_adam_conv_7_bias_m_read_readvariableop0savev2_adam_dense_0_kernel_m_read_readvariableop.savev2_adam_dense_0_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop/savev2_adam_conv_3_kernel_v_read_readvariableop-savev2_adam_conv_3_bias_v_read_readvariableop/savev2_adam_conv_5_kernel_v_read_readvariableop-savev2_adam_conv_5_bias_v_read_readvariableop/savev2_adam_conv_7_kernel_v_read_readvariableop-savev2_adam_conv_7_bias_v_read_readvariableop0savev2_adam_dense_0_kernel_v_read_readvariableop.savev2_adam_dense_0_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *D
dtypes:
826	2
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
�: :d:d:F:F:(:(:	�P:P:PP:P:P<:<:<:: : : : : : : : : : : :d:d:F:F:(:(:	�P:P:PP:P:P<:<:<::d:d:F:F:(:(:	�P:P:PP:P:P<:<:<:: 2(
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:d: 

_output_shapes
:d:($
"
_output_shapes
:F: 

_output_shapes
:F:($
"
_output_shapes
:(: 

_output_shapes
:(:% !

_output_shapes
:	�P: !

_output_shapes
:P:$" 

_output_shapes

:PP: #

_output_shapes
:P:$$ 

_output_shapes

:P<: %

_output_shapes
:<:$& 

_output_shapes

:<: '

_output_shapes
::(($
"
_output_shapes
:d: )

_output_shapes
:d:(*$
"
_output_shapes
:F: +

_output_shapes
:F:(,$
"
_output_shapes
:(: -

_output_shapes
:(:%.!

_output_shapes
:	�P: /

_output_shapes
:P:$0 

_output_shapes

:PP: 1

_output_shapes
:P:$2 

_output_shapes

:P<: 3

_output_shapes
:<:$4 

_output_shapes

:<: 5

_output_shapes
::6

_output_shapes
: 
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_845868

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
�
C
'__inference_pool_3_layer_call_fn_844492

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
GPU2 *0J 8� *K
fFRD
B__inference_pool_3_layer_call_and_return_conditional_losses_8444862
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
��
�
!__inference__wrapped_model_844477
input_onehotP
:model_4_conv_7_conv1d_expanddims_1_readvariableop_resource:(<
.model_4_conv_7_biasadd_readvariableop_resource:(P
:model_4_conv_5_conv1d_expanddims_1_readvariableop_resource:F<
.model_4_conv_5_biasadd_readvariableop_resource:FP
:model_4_conv_3_conv1d_expanddims_1_readvariableop_resource:d<
.model_4_conv_3_biasadd_readvariableop_resource:dA
.model_4_dense_0_matmul_readvariableop_resource:	�P=
/model_4_dense_0_biasadd_readvariableop_resource:P@
.model_4_dense_1_matmul_readvariableop_resource:PP=
/model_4_dense_1_biasadd_readvariableop_resource:P@
.model_4_dense_2_matmul_readvariableop_resource:P<=
/model_4_dense_2_biasadd_readvariableop_resource:<?
-model_4_output_matmul_readvariableop_resource:<<
.model_4_output_biasadd_readvariableop_resource:
identity��%model_4/conv_3/BiasAdd/ReadVariableOp�1model_4/conv_3/conv1d/ExpandDims_1/ReadVariableOp�%model_4/conv_5/BiasAdd/ReadVariableOp�1model_4/conv_5/conv1d/ExpandDims_1/ReadVariableOp�%model_4/conv_7/BiasAdd/ReadVariableOp�1model_4/conv_7/conv1d/ExpandDims_1/ReadVariableOp�&model_4/dense_0/BiasAdd/ReadVariableOp�%model_4/dense_0/MatMul/ReadVariableOp�&model_4/dense_1/BiasAdd/ReadVariableOp�%model_4/dense_1/MatMul/ReadVariableOp�&model_4/dense_2/BiasAdd/ReadVariableOp�%model_4/dense_2/MatMul/ReadVariableOp�%model_4/output/BiasAdd/ReadVariableOp�$model_4/output/MatMul/ReadVariableOp�
$model_4/conv_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2&
$model_4/conv_7/conv1d/ExpandDims/dim�
 model_4/conv_7/conv1d/ExpandDims
ExpandDimsinput_onehot-model_4/conv_7/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2"
 model_4/conv_7/conv1d/ExpandDims�
1model_4/conv_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_4_conv_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:(*
dtype023
1model_4/conv_7/conv1d/ExpandDims_1/ReadVariableOp�
&model_4/conv_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model_4/conv_7/conv1d/ExpandDims_1/dim�
"model_4/conv_7/conv1d/ExpandDims_1
ExpandDims9model_4/conv_7/conv1d/ExpandDims_1/ReadVariableOp:value:0/model_4/conv_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:(2$
"model_4/conv_7/conv1d/ExpandDims_1�
model_4/conv_7/conv1dConv2D)model_4/conv_7/conv1d/ExpandDims:output:0+model_4/conv_7/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������(*
paddingVALID*
strides
2
model_4/conv_7/conv1d�
model_4/conv_7/conv1d/SqueezeSqueezemodel_4/conv_7/conv1d:output:0*
T0*+
_output_shapes
:���������(*
squeeze_dims

���������2
model_4/conv_7/conv1d/Squeeze�
%model_4/conv_7/BiasAdd/ReadVariableOpReadVariableOp.model_4_conv_7_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02'
%model_4/conv_7/BiasAdd/ReadVariableOp�
model_4/conv_7/BiasAddBiasAdd&model_4/conv_7/conv1d/Squeeze:output:0-model_4/conv_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������(2
model_4/conv_7/BiasAdd�
model_4/conv_7/ReluRelumodel_4/conv_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������(2
model_4/conv_7/Relu�
$model_4/conv_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2&
$model_4/conv_5/conv1d/ExpandDims/dim�
 model_4/conv_5/conv1d/ExpandDims
ExpandDimsinput_onehot-model_4/conv_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2"
 model_4/conv_5/conv1d/ExpandDims�
1model_4/conv_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_4_conv_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:F*
dtype023
1model_4/conv_5/conv1d/ExpandDims_1/ReadVariableOp�
&model_4/conv_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model_4/conv_5/conv1d/ExpandDims_1/dim�
"model_4/conv_5/conv1d/ExpandDims_1
ExpandDims9model_4/conv_5/conv1d/ExpandDims_1/ReadVariableOp:value:0/model_4/conv_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:F2$
"model_4/conv_5/conv1d/ExpandDims_1�
model_4/conv_5/conv1dConv2D)model_4/conv_5/conv1d/ExpandDims:output:0+model_4/conv_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������F*
paddingVALID*
strides
2
model_4/conv_5/conv1d�
model_4/conv_5/conv1d/SqueezeSqueezemodel_4/conv_5/conv1d:output:0*
T0*+
_output_shapes
:���������F*
squeeze_dims

���������2
model_4/conv_5/conv1d/Squeeze�
%model_4/conv_5/BiasAdd/ReadVariableOpReadVariableOp.model_4_conv_5_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02'
%model_4/conv_5/BiasAdd/ReadVariableOp�
model_4/conv_5/BiasAddBiasAdd&model_4/conv_5/conv1d/Squeeze:output:0-model_4/conv_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������F2
model_4/conv_5/BiasAdd�
model_4/conv_5/ReluRelumodel_4/conv_5/BiasAdd:output:0*
T0*+
_output_shapes
:���������F2
model_4/conv_5/Relu�
$model_4/conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2&
$model_4/conv_3/conv1d/ExpandDims/dim�
 model_4/conv_3/conv1d/ExpandDims
ExpandDimsinput_onehot-model_4/conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2"
 model_4/conv_3/conv1d/ExpandDims�
1model_4/conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_4_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:d*
dtype023
1model_4/conv_3/conv1d/ExpandDims_1/ReadVariableOp�
&model_4/conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model_4/conv_3/conv1d/ExpandDims_1/dim�
"model_4/conv_3/conv1d/ExpandDims_1
ExpandDims9model_4/conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0/model_4/conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:d2$
"model_4/conv_3/conv1d/ExpandDims_1�
model_4/conv_3/conv1dConv2D)model_4/conv_3/conv1d/ExpandDims:output:0+model_4/conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������d*
paddingVALID*
strides
2
model_4/conv_3/conv1d�
model_4/conv_3/conv1d/SqueezeSqueezemodel_4/conv_3/conv1d:output:0*
T0*+
_output_shapes
:���������d*
squeeze_dims

���������2
model_4/conv_3/conv1d/Squeeze�
%model_4/conv_3/BiasAdd/ReadVariableOpReadVariableOp.model_4_conv_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02'
%model_4/conv_3/BiasAdd/ReadVariableOp�
model_4/conv_3/BiasAddBiasAdd&model_4/conv_3/conv1d/Squeeze:output:0-model_4/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������d2
model_4/conv_3/BiasAdd�
model_4/conv_3/ReluRelumodel_4/conv_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������d2
model_4/conv_3/Relu�
model_4/drop_7/IdentityIdentity!model_4/conv_7/Relu:activations:0*
T0*+
_output_shapes
:���������(2
model_4/drop_7/Identity�
model_4/drop_5/IdentityIdentity!model_4/conv_5/Relu:activations:0*
T0*+
_output_shapes
:���������F2
model_4/drop_5/Identity�
model_4/drop_3/IdentityIdentity!model_4/conv_3/Relu:activations:0*
T0*+
_output_shapes
:���������d2
model_4/drop_3/Identity�
model_4/pool_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
model_4/pool_7/ExpandDims/dim�
model_4/pool_7/ExpandDims
ExpandDims model_4/drop_7/Identity:output:0&model_4/pool_7/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������(2
model_4/pool_7/ExpandDims�
model_4/pool_7/AvgPoolAvgPool"model_4/pool_7/ExpandDims:output:0*
T0*/
_output_shapes
:���������	(*
ksize
*
paddingSAME*
strides
2
model_4/pool_7/AvgPool�
model_4/pool_7/SqueezeSqueezemodel_4/pool_7/AvgPool:output:0*
T0*+
_output_shapes
:���������	(*
squeeze_dims
2
model_4/pool_7/Squeeze�
model_4/pool_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
model_4/pool_5/ExpandDims/dim�
model_4/pool_5/ExpandDims
ExpandDims model_4/drop_5/Identity:output:0&model_4/pool_5/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������F2
model_4/pool_5/ExpandDims�
model_4/pool_5/AvgPoolAvgPool"model_4/pool_5/ExpandDims:output:0*
T0*/
_output_shapes
:���������
F*
ksize
*
paddingSAME*
strides
2
model_4/pool_5/AvgPool�
model_4/pool_5/SqueezeSqueezemodel_4/pool_5/AvgPool:output:0*
T0*+
_output_shapes
:���������
F*
squeeze_dims
2
model_4/pool_5/Squeeze�
model_4/pool_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
model_4/pool_3/ExpandDims/dim�
model_4/pool_3/ExpandDims
ExpandDims model_4/drop_3/Identity:output:0&model_4/pool_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d2
model_4/pool_3/ExpandDims�
model_4/pool_3/AvgPoolAvgPool"model_4/pool_3/ExpandDims:output:0*
T0*/
_output_shapes
:���������d*
ksize
*
paddingSAME*
strides
2
model_4/pool_3/AvgPool�
model_4/pool_3/SqueezeSqueezemodel_4/pool_3/AvgPool:output:0*
T0*+
_output_shapes
:���������d*
squeeze_dims
2
model_4/pool_3/Squeeze�
model_4/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����L  2
model_4/flatten_3/Const�
model_4/flatten_3/ReshapeReshapemodel_4/pool_3/Squeeze:output:0 model_4/flatten_3/Const:output:0*
T0*(
_output_shapes
:����������2
model_4/flatten_3/Reshape�
model_4/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
model_4/flatten_5/Const�
model_4/flatten_5/ReshapeReshapemodel_4/pool_5/Squeeze:output:0 model_4/flatten_5/Const:output:0*
T0*(
_output_shapes
:����������2
model_4/flatten_5/Reshape�
model_4/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"����h  2
model_4/flatten_7/Const�
model_4/flatten_7/ReshapeReshapemodel_4/pool_7/Squeeze:output:0 model_4/flatten_7/Const:output:0*
T0*(
_output_shapes
:����������2
model_4/flatten_7/Reshape�
!model_4/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_4/concatenate_4/concat/axis�
model_4/concatenate_4/concatConcatV2"model_4/flatten_3/Reshape:output:0"model_4/flatten_5/Reshape:output:0"model_4/flatten_7/Reshape:output:0*model_4/concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
model_4/concatenate_4/concat�
%model_4/dense_0/MatMul/ReadVariableOpReadVariableOp.model_4_dense_0_matmul_readvariableop_resource*
_output_shapes
:	�P*
dtype02'
%model_4/dense_0/MatMul/ReadVariableOp�
model_4/dense_0/MatMulMatMul%model_4/concatenate_4/concat:output:0-model_4/dense_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
model_4/dense_0/MatMul�
&model_4/dense_0/BiasAdd/ReadVariableOpReadVariableOp/model_4_dense_0_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02(
&model_4/dense_0/BiasAdd/ReadVariableOp�
model_4/dense_0/BiasAddBiasAdd model_4/dense_0/MatMul:product:0.model_4/dense_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
model_4/dense_0/BiasAdd�
model_4/dense_0/ReluRelu model_4/dense_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
model_4/dense_0/Relu�
model_4/drop_d0/IdentityIdentity"model_4/dense_0/Relu:activations:0*
T0*'
_output_shapes
:���������P2
model_4/drop_d0/Identity�
%model_4/dense_1/MatMul/ReadVariableOpReadVariableOp.model_4_dense_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02'
%model_4/dense_1/MatMul/ReadVariableOp�
model_4/dense_1/MatMulMatMul!model_4/drop_d0/Identity:output:0-model_4/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
model_4/dense_1/MatMul�
&model_4/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_4_dense_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02(
&model_4/dense_1/BiasAdd/ReadVariableOp�
model_4/dense_1/BiasAddBiasAdd model_4/dense_1/MatMul:product:0.model_4/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
model_4/dense_1/BiasAdd�
model_4/dense_1/ReluRelu model_4/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
model_4/dense_1/Relu�
model_4/drop_d1/IdentityIdentity"model_4/dense_1/Relu:activations:0*
T0*'
_output_shapes
:���������P2
model_4/drop_d1/Identity�
%model_4/dense_2/MatMul/ReadVariableOpReadVariableOp.model_4_dense_2_matmul_readvariableop_resource*
_output_shapes

:P<*
dtype02'
%model_4/dense_2/MatMul/ReadVariableOp�
model_4/dense_2/MatMulMatMul!model_4/drop_d1/Identity:output:0-model_4/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
model_4/dense_2/MatMul�
&model_4/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_4_dense_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02(
&model_4/dense_2/BiasAdd/ReadVariableOp�
model_4/dense_2/BiasAddBiasAdd model_4/dense_2/MatMul:product:0.model_4/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
model_4/dense_2/BiasAdd�
model_4/dense_2/ReluRelu model_4/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������<2
model_4/dense_2/Relu�
model_4/drop_d2/IdentityIdentity"model_4/dense_2/Relu:activations:0*
T0*'
_output_shapes
:���������<2
model_4/drop_d2/Identity�
$model_4/output/MatMul/ReadVariableOpReadVariableOp-model_4_output_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02&
$model_4/output/MatMul/ReadVariableOp�
model_4/output/MatMulMatMul!model_4/drop_d2/Identity:output:0,model_4/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_4/output/MatMul�
%model_4/output/BiasAdd/ReadVariableOpReadVariableOp.model_4_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model_4/output/BiasAdd/ReadVariableOp�
model_4/output/BiasAddBiasAddmodel_4/output/MatMul:product:0-model_4/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_4/output/BiasAdd�
IdentityIdentitymodel_4/output/BiasAdd:output:0&^model_4/conv_3/BiasAdd/ReadVariableOp2^model_4/conv_3/conv1d/ExpandDims_1/ReadVariableOp&^model_4/conv_5/BiasAdd/ReadVariableOp2^model_4/conv_5/conv1d/ExpandDims_1/ReadVariableOp&^model_4/conv_7/BiasAdd/ReadVariableOp2^model_4/conv_7/conv1d/ExpandDims_1/ReadVariableOp'^model_4/dense_0/BiasAdd/ReadVariableOp&^model_4/dense_0/MatMul/ReadVariableOp'^model_4/dense_1/BiasAdd/ReadVariableOp&^model_4/dense_1/MatMul/ReadVariableOp'^model_4/dense_2/BiasAdd/ReadVariableOp&^model_4/dense_2/MatMul/ReadVariableOp&^model_4/output/BiasAdd/ReadVariableOp%^model_4/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������: : : : : : : : : : : : : : 2N
%model_4/conv_3/BiasAdd/ReadVariableOp%model_4/conv_3/BiasAdd/ReadVariableOp2f
1model_4/conv_3/conv1d/ExpandDims_1/ReadVariableOp1model_4/conv_3/conv1d/ExpandDims_1/ReadVariableOp2N
%model_4/conv_5/BiasAdd/ReadVariableOp%model_4/conv_5/BiasAdd/ReadVariableOp2f
1model_4/conv_5/conv1d/ExpandDims_1/ReadVariableOp1model_4/conv_5/conv1d/ExpandDims_1/ReadVariableOp2N
%model_4/conv_7/BiasAdd/ReadVariableOp%model_4/conv_7/BiasAdd/ReadVariableOp2f
1model_4/conv_7/conv1d/ExpandDims_1/ReadVariableOp1model_4/conv_7/conv1d/ExpandDims_1/ReadVariableOp2P
&model_4/dense_0/BiasAdd/ReadVariableOp&model_4/dense_0/BiasAdd/ReadVariableOp2N
%model_4/dense_0/MatMul/ReadVariableOp%model_4/dense_0/MatMul/ReadVariableOp2P
&model_4/dense_1/BiasAdd/ReadVariableOp&model_4/dense_1/BiasAdd/ReadVariableOp2N
%model_4/dense_1/MatMul/ReadVariableOp%model_4/dense_1/MatMul/ReadVariableOp2P
&model_4/dense_2/BiasAdd/ReadVariableOp&model_4/dense_2/BiasAdd/ReadVariableOp2N
%model_4/dense_2/MatMul/ReadVariableOp%model_4/dense_2/MatMul/ReadVariableOp2N
%model_4/output/BiasAdd/ReadVariableOp%model_4/output/BiasAdd/ReadVariableOp2L
$model_4/output/MatMul/ReadVariableOp$model_4/output/MatMul/ReadVariableOp:Y U
+
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�
C
'__inference_drop_5_layer_call_fn_845704

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
GPU2 *0J 8� *K
fFRD
B__inference_drop_5_layer_call_and_return_conditional_losses_8446072
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
�
C
'__inference_pool_5_layer_call_fn_844507

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
GPU2 *0J 8� *K
fFRD
B__inference_pool_5_layer_call_and_return_conditional_losses_8445012
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
�
D
(__inference_drop_d2_layer_call_fn_845920

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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d2_layer_call_and_return_conditional_losses_8447232
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
�
b
C__inference_drop_d2_layer_call_and_return_conditional_losses_844803

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
�U
�
C__inference_model_4_layer_call_and_return_conditional_losses_845094

inputs#
conv_7_845045:(
conv_7_845047:(#
conv_5_845050:F
conv_5_845052:F#
conv_3_845055:d
conv_3_845057:d!
dense_0_845070:	�P
dense_0_845072:P 
dense_1_845076:PP
dense_1_845078:P 
dense_2_845082:P<
dense_2_845084:<
output_845088:<
output_845090:
identity��conv_3/StatefulPartitionedCall�conv_5/StatefulPartitionedCall�conv_7/StatefulPartitionedCall�dense_0/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�drop_3/StatefulPartitionedCall�drop_5/StatefulPartitionedCall�drop_7/StatefulPartitionedCall�drop_d0/StatefulPartitionedCall�drop_d1/StatefulPartitionedCall�drop_d2/StatefulPartitionedCall�output/StatefulPartitionedCall�
conv_7/StatefulPartitionedCallStatefulPartitionedCallinputsconv_7_845045conv_7_845047*
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
GPU2 *0J 8� *K
fFRD
B__inference_conv_7_layer_call_and_return_conditional_losses_8445452 
conv_7/StatefulPartitionedCall�
conv_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv_5_845050conv_5_845052*
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
GPU2 *0J 8� *K
fFRD
B__inference_conv_5_layer_call_and_return_conditional_losses_8445672 
conv_5/StatefulPartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv_3_845055conv_3_845057*
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
GPU2 *0J 8� *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_8445892 
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
GPU2 *0J 8� *K
fFRD
B__inference_drop_7_layer_call_and_return_conditional_losses_8449742 
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
GPU2 *0J 8� *K
fFRD
B__inference_drop_5_layer_call_and_return_conditional_losses_8449512 
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
GPU2 *0J 8� *K
fFRD
B__inference_drop_3_layer_call_and_return_conditional_losses_8449282 
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
GPU2 *0J 8� *K
fFRD
B__inference_pool_7_layer_call_and_return_conditional_losses_8445162
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
GPU2 *0J 8� *K
fFRD
B__inference_pool_5_layer_call_and_return_conditional_losses_8445012
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
GPU2 *0J 8� *K
fFRD
B__inference_pool_3_layer_call_and_return_conditional_losses_8444862
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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_8446252
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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_8446332
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
GPU2 *0J 8� *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_8446412
flatten_7/PartitionedCall�
concatenate_4/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0*
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
GPU2 *0J 8� *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_8446512
concatenate_4/PartitionedCall�
dense_0/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_0_845070dense_0_845072*
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
GPU2 *0J 8� *L
fGRE
C__inference_dense_0_layer_call_and_return_conditional_losses_8446642!
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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d0_layer_call_and_return_conditional_losses_8448692!
drop_d0/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(drop_d0/StatefulPartitionedCall:output:0dense_1_845076dense_1_845078*
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
GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_8446882!
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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d1_layer_call_and_return_conditional_losses_8448362!
drop_d1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(drop_d1/StatefulPartitionedCall:output:0dense_2_845082dense_2_845084*
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
GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_8447122!
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
GPU2 *0J 8� *L
fGRE
C__inference_drop_d2_layer_call_and_return_conditional_losses_8448032!
drop_d2/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall(drop_d2/StatefulPartitionedCall:output:0output_845088output_845090*
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
GPU2 *0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_8447352 
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
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_844688

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
�
b
C__inference_drop_d2_layer_call_and_return_conditional_losses_845942

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
a
C__inference_drop_d0_layer_call_and_return_conditional_losses_845836

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
�
�
B__inference_conv_5_layer_call_and_return_conditional_losses_844567

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
a
B__inference_drop_5_layer_call_and_return_conditional_losses_845726

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
�
`
B__inference_drop_7_layer_call_and_return_conditional_losses_845741

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
�
�
B__inference_conv_7_layer_call_and_return_conditional_losses_845672

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
�
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_845764

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
�
a
E__inference_flatten_7_layer_call_and_return_conditional_losses_845786

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
��
�
C__inference_model_4_layer_call_and_return_conditional_losses_845597

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
flatten_7/Reshapex
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_4/concat/axis�
concatenate_4/concatConcatV2flatten_3/Reshape:output:0flatten_5/Reshape:output:0flatten_7/Reshape:output:0"concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatenate_4/concat�
dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes
:	�P*
dtype02
dense_0/MatMul/ReadVariableOp�
dense_0/MatMulMatMulconcatenate_4/concat:output:0%dense_0/MatMul/ReadVariableOp:value:0*
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
`
B__inference_drop_5_layer_call_and_return_conditional_losses_845714

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
�

�
C__inference_dense_0_layer_call_and_return_conditional_losses_844664

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
�
^
B__inference_pool_7_layer_call_and_return_conditional_losses_844516

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
�
^
B__inference_pool_5_layer_call_and_return_conditional_losses_844501

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
�
�
'__inference_conv_7_layer_call_fn_845656

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
GPU2 *0J 8� *K
fFRD
B__inference_conv_7_layer_call_and_return_conditional_losses_8445452
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
�
(__inference_model_4_layer_call_fn_845158
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
GPU2 *0J 8� *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_8450942
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
_user_specified_nameinput_onehot"�L
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
regularization_losses
trainable_variables
	variables
	keras_api

signatures
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses"��
_tf_keras_networkÊ{"name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 100, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["input_onehot", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 70, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_5", "inbound_nodes": [[["input_onehot", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 40, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_7", "inbound_nodes": [[["input_onehot", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_3", "inbound_nodes": [[["conv_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_5", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_5", "inbound_nodes": [[["conv_5", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_7", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_7", "inbound_nodes": [[["conv_7", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "pool_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "name": "pool_3", "inbound_nodes": [[["drop_3", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "pool_5", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "name": "pool_5", "inbound_nodes": [[["drop_5", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "pool_7", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "name": "pool_7", "inbound_nodes": [[["drop_7", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["pool_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["pool_5", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["pool_7", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_4", "inbound_nodes": [[["flatten_3", 0, 0, {}], ["flatten_5", 0, 0, {}], ["flatten_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_0", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_0", "inbound_nodes": [[["concatenate_4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_d0", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_d0", "inbound_nodes": [[["dense_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["drop_d0", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_d1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_d1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["drop_d1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_d2", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_d2", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["drop_d2", 0, 0, {}]]]}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 35, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 4]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 23, 4]}, "float32", "input_onehot"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "conv_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 100, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Conv1D", "config": {"name": "conv_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 70, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_5", "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Conv1D", "config": {"name": "conv_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 40, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_7", "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "drop_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_3", "inbound_nodes": [[["conv_3", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dropout", "config": {"name": "drop_5", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_5", "inbound_nodes": [[["conv_5", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dropout", "config": {"name": "drop_7", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_7", "inbound_nodes": [[["conv_7", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "AveragePooling1D", "config": {"name": "pool_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "name": "pool_3", "inbound_nodes": [[["drop_3", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "AveragePooling1D", "config": {"name": "pool_5", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "name": "pool_5", "inbound_nodes": [[["drop_5", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "AveragePooling1D", "config": {"name": "pool_7", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "name": "pool_7", "inbound_nodes": [[["drop_7", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["pool_3", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["pool_5", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["pool_7", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Concatenate", "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_4", "inbound_nodes": [[["flatten_3", 0, 0, {}], ["flatten_5", 0, 0, {}], ["flatten_7", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "Dense", "config": {"name": "dense_0", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_0", "inbound_nodes": [[["concatenate_4", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Dropout", "config": {"name": "drop_d0", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_d0", "inbound_nodes": [[["dense_0", 0, 0, {}]]], "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["drop_d0", 0, 0, {}]]], "shared_object_id": 26}, {"class_name": "Dropout", "config": {"name": "drop_d1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_d1", "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 27}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["drop_d1", 0, 0, {}]]], "shared_object_id": 30}, {"class_name": "Dropout", "config": {"name": "drop_d2", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_d2", "inbound_nodes": [[["dense_2", 0, 0, {}]]], "shared_object_id": 31}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["drop_d2", 0, 0, {}]]], "shared_object_id": 34}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 37}, {"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}, "shared_object_id": 38}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_onehot", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}}
�

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�

_tf_keras_layer�
{"name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 100, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 4}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 4]}}
�

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�

_tf_keras_layer�
{"name": "conv_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 70, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 4}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 4]}}
�

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�

_tf_keras_layer�
{"name": "conv_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 40, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 4}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 4]}}
�
.regularization_losses
/trainable_variables
0	variables
1	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "drop_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["conv_3", 0, 0, {}]]], "shared_object_id": 10}
�
2regularization_losses
3trainable_variables
4	variables
5	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "drop_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop_5", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["conv_5", 0, 0, {}]]], "shared_object_id": 11}
�
6regularization_losses
7trainable_variables
8	variables
9	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "drop_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop_7", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["conv_7", 0, 0, {}]]], "shared_object_id": 12}
�
:regularization_losses
;trainable_variables
<	variables
=	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "pool_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AveragePooling1D", "config": {"name": "pool_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "inbound_nodes": [[["drop_3", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 42}}
�
>regularization_losses
?trainable_variables
@	variables
A	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "pool_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AveragePooling1D", "config": {"name": "pool_5", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "inbound_nodes": [[["drop_5", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 43}}
�
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "pool_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AveragePooling1D", "config": {"name": "pool_7", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "inbound_nodes": [[["drop_7", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 44}}
�
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["pool_3", 0, 0, {}]]], "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 45}}
�
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["pool_5", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 46}}
�
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "flatten_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["pool_7", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 47}}
�
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "concatenate_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["flatten_3", 0, 0, {}], ["flatten_5", 0, 0, {}], ["flatten_7", 0, 0, {}]]], "shared_object_id": 19, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1100]}, {"class_name": "TensorShape", "items": [null, 700]}, {"class_name": "TensorShape", "items": [null, 360]}]}
�	

Vkernel
Wbias
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dense_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_0", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate_4", 0, 0, {}]]], "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2160}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2160]}}
�
\regularization_losses
]trainable_variables
^	variables
_	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "drop_d0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop_d0", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_0", 0, 0, {}]]], "shared_object_id": 23}
�

`kernel
abias
bregularization_losses
ctrainable_variables
d	variables
e	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["drop_d0", 0, 0, {}]]], "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
�
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "drop_d1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop_d1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 27}
�

jkernel
kbias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["drop_d1", 0, 0, {}]]], "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}, "shared_object_id": 50}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
�
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "drop_d2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop_d2", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_2", 0, 0, {}]]], "shared_object_id": 31}
�

tkernel
ubias
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["drop_d2", 0, 0, {}]]], "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}, "shared_object_id": 51}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
�
ziter

{beta_1

|beta_2
	}decay
~learning_ratem�m�"m�#m�(m�)m�Vm�Wm�`m�am�jm�km�tm�um�v�v�"v�#v�(v�)v�Vv�Wv�`v�av�jv�kv�tv�uv�"
	optimizer
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
regularization_losses
�non_trainable_variables
trainable_variables
�metrics
 �layer_regularization_losses
	variables
�layers
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
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
�layer_metrics
regularization_losses
�non_trainable_variables
trainable_variables
�metrics
 �layer_regularization_losses
 	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!F2conv_5/kernel
:F2conv_5/bias
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
�
�layer_metrics
$regularization_losses
�non_trainable_variables
%trainable_variables
�metrics
 �layer_regularization_losses
&	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!(2conv_7/kernel
:(2conv_7/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
�
�layer_metrics
*regularization_losses
�non_trainable_variables
+trainable_variables
�metrics
 �layer_regularization_losses
,	variables
�layers
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
�layer_metrics
.regularization_losses
�non_trainable_variables
/trainable_variables
�metrics
 �layer_regularization_losses
0	variables
�layers
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
�layer_metrics
2regularization_losses
�non_trainable_variables
3trainable_variables
�metrics
 �layer_regularization_losses
4	variables
�layers
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
�layer_metrics
6regularization_losses
�non_trainable_variables
7trainable_variables
�metrics
 �layer_regularization_losses
8	variables
�layers
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
�layer_metrics
:regularization_losses
�non_trainable_variables
;trainable_variables
�metrics
 �layer_regularization_losses
<	variables
�layers
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
�layer_metrics
>regularization_losses
�non_trainable_variables
?trainable_variables
�metrics
 �layer_regularization_losses
@	variables
�layers
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
�layer_metrics
Bregularization_losses
�non_trainable_variables
Ctrainable_variables
�metrics
 �layer_regularization_losses
D	variables
�layers
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
�layer_metrics
Fregularization_losses
�non_trainable_variables
Gtrainable_variables
�metrics
 �layer_regularization_losses
H	variables
�layers
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
�layer_metrics
Jregularization_losses
�non_trainable_variables
Ktrainable_variables
�metrics
 �layer_regularization_losses
L	variables
�layers
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
�layer_metrics
Nregularization_losses
�non_trainable_variables
Otrainable_variables
�metrics
 �layer_regularization_losses
P	variables
�layers
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
�layer_metrics
Rregularization_losses
�non_trainable_variables
Strainable_variables
�metrics
 �layer_regularization_losses
T	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	�P2dense_0/kernel
:P2dense_0/bias
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
�
�layer_metrics
Xregularization_losses
�non_trainable_variables
Ytrainable_variables
�metrics
 �layer_regularization_losses
Z	variables
�layers
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
�layer_metrics
\regularization_losses
�non_trainable_variables
]trainable_variables
�metrics
 �layer_regularization_losses
^	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :PP2dense_1/kernel
:P2dense_1/bias
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
�
�layer_metrics
bregularization_losses
�non_trainable_variables
ctrainable_variables
�metrics
 �layer_regularization_losses
d	variables
�layers
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
�layer_metrics
fregularization_losses
�non_trainable_variables
gtrainable_variables
�metrics
 �layer_regularization_losses
h	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :P<2dense_2/kernel
:<2dense_2/bias
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
�
�layer_metrics
lregularization_losses
�non_trainable_variables
mtrainable_variables
�metrics
 �layer_regularization_losses
n	variables
�layers
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
�layer_metrics
pregularization_losses
�non_trainable_variables
qtrainable_variables
�metrics
 �layer_regularization_losses
r	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:<2output/kernel
:2output/bias
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
�
�layer_metrics
vregularization_losses
�non_trainable_variables
wtrainable_variables
�metrics
 �layer_regularization_losses
x	variables
�layers
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
 "
trackable_list_wrapper
8
�0
�1
�2"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 52}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 37}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}, "shared_object_id": 38}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
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
�2�
(__inference_model_4_layer_call_fn_844773
(__inference_model_4_layer_call_fn_845336
(__inference_model_4_layer_call_fn_845369
(__inference_model_4_layer_call_fn_845158�
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
!__inference__wrapped_model_844477�
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
�2�
C__inference_model_4_layer_call_and_return_conditional_losses_845462
C__inference_model_4_layer_call_and_return_conditional_losses_845597
C__inference_model_4_layer_call_and_return_conditional_losses_845210
C__inference_model_4_layer_call_and_return_conditional_losses_845262�
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
'__inference_conv_3_layer_call_fn_845606�
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
B__inference_conv_3_layer_call_and_return_conditional_losses_845622�
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
'__inference_conv_5_layer_call_fn_845631�
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
B__inference_conv_5_layer_call_and_return_conditional_losses_845647�
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
'__inference_conv_7_layer_call_fn_845656�
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
B__inference_conv_7_layer_call_and_return_conditional_losses_845672�
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
'__inference_drop_3_layer_call_fn_845677
'__inference_drop_3_layer_call_fn_845682�
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
B__inference_drop_3_layer_call_and_return_conditional_losses_845687
B__inference_drop_3_layer_call_and_return_conditional_losses_845699�
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
'__inference_drop_5_layer_call_fn_845704
'__inference_drop_5_layer_call_fn_845709�
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
B__inference_drop_5_layer_call_and_return_conditional_losses_845714
B__inference_drop_5_layer_call_and_return_conditional_losses_845726�
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
'__inference_drop_7_layer_call_fn_845731
'__inference_drop_7_layer_call_fn_845736�
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
B__inference_drop_7_layer_call_and_return_conditional_losses_845741
B__inference_drop_7_layer_call_and_return_conditional_losses_845753�
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
�2�
'__inference_pool_3_layer_call_fn_844492�
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
B__inference_pool_3_layer_call_and_return_conditional_losses_844486�
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
�2�
'__inference_pool_5_layer_call_fn_844507�
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
B__inference_pool_5_layer_call_and_return_conditional_losses_844501�
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
�2�
'__inference_pool_7_layer_call_fn_844522�
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
B__inference_pool_7_layer_call_and_return_conditional_losses_844516�
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
*__inference_flatten_3_layer_call_fn_845758�
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
E__inference_flatten_3_layer_call_and_return_conditional_losses_845764�
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
*__inference_flatten_5_layer_call_fn_845769�
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_845775�
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
*__inference_flatten_7_layer_call_fn_845780�
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
E__inference_flatten_7_layer_call_and_return_conditional_losses_845786�
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
.__inference_concatenate_4_layer_call_fn_845793�
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
I__inference_concatenate_4_layer_call_and_return_conditional_losses_845801�
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
(__inference_dense_0_layer_call_fn_845810�
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
C__inference_dense_0_layer_call_and_return_conditional_losses_845821�
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
(__inference_drop_d0_layer_call_fn_845826
(__inference_drop_d0_layer_call_fn_845831�
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
C__inference_drop_d0_layer_call_and_return_conditional_losses_845836
C__inference_drop_d0_layer_call_and_return_conditional_losses_845848�
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
(__inference_dense_1_layer_call_fn_845857�
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
C__inference_dense_1_layer_call_and_return_conditional_losses_845868�
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
(__inference_drop_d1_layer_call_fn_845873
(__inference_drop_d1_layer_call_fn_845878�
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
C__inference_drop_d1_layer_call_and_return_conditional_losses_845883
C__inference_drop_d1_layer_call_and_return_conditional_losses_845895�
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
(__inference_dense_2_layer_call_fn_845904�
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
C__inference_dense_2_layer_call_and_return_conditional_losses_845915�
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
(__inference_drop_d2_layer_call_fn_845920
(__inference_drop_d2_layer_call_fn_845925�
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
C__inference_drop_d2_layer_call_and_return_conditional_losses_845930
C__inference_drop_d2_layer_call_and_return_conditional_losses_845942�
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
'__inference_output_layer_call_fn_845951�
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
B__inference_output_layer_call_and_return_conditional_losses_845961�
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
$__inference_signature_wrapper_845303input_onehot"�
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
!__inference__wrapped_model_844477|()"#VW`ajktu9�6
/�,
*�'
input_onehot���������
� "/�,
*
output �
output����������
I__inference_concatenate_4_layer_call_and_return_conditional_losses_845801���~
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
.__inference_concatenate_4_layer_call_fn_845793���~
w�t
r�o
#� 
inputs/0����������
#� 
inputs/1����������
#� 
inputs/2����������
� "������������
B__inference_conv_3_layer_call_and_return_conditional_losses_845622d3�0
)�&
$�!
inputs���������
� ")�&
�
0���������d
� �
'__inference_conv_3_layer_call_fn_845606W3�0
)�&
$�!
inputs���������
� "����������d�
B__inference_conv_5_layer_call_and_return_conditional_losses_845647d"#3�0
)�&
$�!
inputs���������
� ")�&
�
0���������F
� �
'__inference_conv_5_layer_call_fn_845631W"#3�0
)�&
$�!
inputs���������
� "����������F�
B__inference_conv_7_layer_call_and_return_conditional_losses_845672d()3�0
)�&
$�!
inputs���������
� ")�&
�
0���������(
� �
'__inference_conv_7_layer_call_fn_845656W()3�0
)�&
$�!
inputs���������
� "����������(�
C__inference_dense_0_layer_call_and_return_conditional_losses_845821]VW0�-
&�#
!�
inputs����������
� "%�"
�
0���������P
� |
(__inference_dense_0_layer_call_fn_845810PVW0�-
&�#
!�
inputs����������
� "����������P�
C__inference_dense_1_layer_call_and_return_conditional_losses_845868\`a/�,
%�"
 �
inputs���������P
� "%�"
�
0���������P
� {
(__inference_dense_1_layer_call_fn_845857O`a/�,
%�"
 �
inputs���������P
� "����������P�
C__inference_dense_2_layer_call_and_return_conditional_losses_845915\jk/�,
%�"
 �
inputs���������P
� "%�"
�
0���������<
� {
(__inference_dense_2_layer_call_fn_845904Ojk/�,
%�"
 �
inputs���������P
� "����������<�
B__inference_drop_3_layer_call_and_return_conditional_losses_845687d7�4
-�*
$�!
inputs���������d
p 
� ")�&
�
0���������d
� �
B__inference_drop_3_layer_call_and_return_conditional_losses_845699d7�4
-�*
$�!
inputs���������d
p
� ")�&
�
0���������d
� �
'__inference_drop_3_layer_call_fn_845677W7�4
-�*
$�!
inputs���������d
p 
� "����������d�
'__inference_drop_3_layer_call_fn_845682W7�4
-�*
$�!
inputs���������d
p
� "����������d�
B__inference_drop_5_layer_call_and_return_conditional_losses_845714d7�4
-�*
$�!
inputs���������F
p 
� ")�&
�
0���������F
� �
B__inference_drop_5_layer_call_and_return_conditional_losses_845726d7�4
-�*
$�!
inputs���������F
p
� ")�&
�
0���������F
� �
'__inference_drop_5_layer_call_fn_845704W7�4
-�*
$�!
inputs���������F
p 
� "����������F�
'__inference_drop_5_layer_call_fn_845709W7�4
-�*
$�!
inputs���������F
p
� "����������F�
B__inference_drop_7_layer_call_and_return_conditional_losses_845741d7�4
-�*
$�!
inputs���������(
p 
� ")�&
�
0���������(
� �
B__inference_drop_7_layer_call_and_return_conditional_losses_845753d7�4
-�*
$�!
inputs���������(
p
� ")�&
�
0���������(
� �
'__inference_drop_7_layer_call_fn_845731W7�4
-�*
$�!
inputs���������(
p 
� "����������(�
'__inference_drop_7_layer_call_fn_845736W7�4
-�*
$�!
inputs���������(
p
� "����������(�
C__inference_drop_d0_layer_call_and_return_conditional_losses_845836\3�0
)�&
 �
inputs���������P
p 
� "%�"
�
0���������P
� �
C__inference_drop_d0_layer_call_and_return_conditional_losses_845848\3�0
)�&
 �
inputs���������P
p
� "%�"
�
0���������P
� {
(__inference_drop_d0_layer_call_fn_845826O3�0
)�&
 �
inputs���������P
p 
� "����������P{
(__inference_drop_d0_layer_call_fn_845831O3�0
)�&
 �
inputs���������P
p
� "����������P�
C__inference_drop_d1_layer_call_and_return_conditional_losses_845883\3�0
)�&
 �
inputs���������P
p 
� "%�"
�
0���������P
� �
C__inference_drop_d1_layer_call_and_return_conditional_losses_845895\3�0
)�&
 �
inputs���������P
p
� "%�"
�
0���������P
� {
(__inference_drop_d1_layer_call_fn_845873O3�0
)�&
 �
inputs���������P
p 
� "����������P{
(__inference_drop_d1_layer_call_fn_845878O3�0
)�&
 �
inputs���������P
p
� "����������P�
C__inference_drop_d2_layer_call_and_return_conditional_losses_845930\3�0
)�&
 �
inputs���������<
p 
� "%�"
�
0���������<
� �
C__inference_drop_d2_layer_call_and_return_conditional_losses_845942\3�0
)�&
 �
inputs���������<
p
� "%�"
�
0���������<
� {
(__inference_drop_d2_layer_call_fn_845920O3�0
)�&
 �
inputs���������<
p 
� "����������<{
(__inference_drop_d2_layer_call_fn_845925O3�0
)�&
 �
inputs���������<
p
� "����������<�
E__inference_flatten_3_layer_call_and_return_conditional_losses_845764]3�0
)�&
$�!
inputs���������d
� "&�#
�
0����������
� ~
*__inference_flatten_3_layer_call_fn_845758P3�0
)�&
$�!
inputs���������d
� "������������
E__inference_flatten_5_layer_call_and_return_conditional_losses_845775]3�0
)�&
$�!
inputs���������
F
� "&�#
�
0����������
� ~
*__inference_flatten_5_layer_call_fn_845769P3�0
)�&
$�!
inputs���������
F
� "������������
E__inference_flatten_7_layer_call_and_return_conditional_losses_845786]3�0
)�&
$�!
inputs���������	(
� "&�#
�
0����������
� ~
*__inference_flatten_7_layer_call_fn_845780P3�0
)�&
$�!
inputs���������	(
� "������������
C__inference_model_4_layer_call_and_return_conditional_losses_845210z()"#VW`ajktuA�>
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
C__inference_model_4_layer_call_and_return_conditional_losses_845262z()"#VW`ajktuA�>
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
C__inference_model_4_layer_call_and_return_conditional_losses_845462t()"#VW`ajktu;�8
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
C__inference_model_4_layer_call_and_return_conditional_losses_845597t()"#VW`ajktu;�8
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
(__inference_model_4_layer_call_fn_844773m()"#VW`ajktuA�>
7�4
*�'
input_onehot���������
p 

 
� "�����������
(__inference_model_4_layer_call_fn_845158m()"#VW`ajktuA�>
7�4
*�'
input_onehot���������
p

 
� "�����������
(__inference_model_4_layer_call_fn_845336g()"#VW`ajktu;�8
1�.
$�!
inputs���������
p 

 
� "�����������
(__inference_model_4_layer_call_fn_845369g()"#VW`ajktu;�8
1�.
$�!
inputs���������
p

 
� "�����������
B__inference_output_layer_call_and_return_conditional_losses_845961\tu/�,
%�"
 �
inputs���������<
� "%�"
�
0���������
� z
'__inference_output_layer_call_fn_845951Otu/�,
%�"
 �
inputs���������<
� "�����������
B__inference_pool_3_layer_call_and_return_conditional_losses_844486�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
'__inference_pool_3_layer_call_fn_844492wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
B__inference_pool_5_layer_call_and_return_conditional_losses_844501�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
'__inference_pool_5_layer_call_fn_844507wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
B__inference_pool_7_layer_call_and_return_conditional_losses_844516�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
'__inference_pool_7_layer_call_fn_844522wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
$__inference_signature_wrapper_845303�()"#VW`ajktuI�F
� 
?�<
:
input_onehot*�'
input_onehot���������"/�,
*
output �
output���������