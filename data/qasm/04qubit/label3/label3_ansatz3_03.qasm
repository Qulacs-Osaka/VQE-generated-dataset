OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.9130714973240925) q[0];
rz(0.2934279785689551) q[0];
ry(-1.648666721922153) q[1];
rz(-0.528094042794044) q[1];
ry(2.064238900326535) q[2];
rz(-0.526935818643291) q[2];
ry(1.6406381527084146) q[3];
rz(-2.5560521667867326) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.6258621078033855) q[0];
rz(-2.799713738598187) q[0];
ry(-0.4366598064093654) q[1];
rz(-0.3825094589346789) q[1];
ry(-2.086147756253097) q[2];
rz(-1.2812757294091028) q[2];
ry(0.04793532423318615) q[3];
rz(-1.9918604838710088) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.7692573090640843) q[0];
rz(0.3902161604054122) q[0];
ry(-1.3952140737076764) q[1];
rz(-2.975881090855021) q[1];
ry(-1.473161881969437) q[2];
rz(-0.2407585142224866) q[2];
ry(0.5757106650559702) q[3];
rz(2.2395925483787305) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.933532283549932) q[0];
rz(-0.5431477727879859) q[0];
ry(-0.7838324326975075) q[1];
rz(-0.19868060574265375) q[1];
ry(-0.23997695082213047) q[2];
rz(-2.5909604891871236) q[2];
ry(-0.8917845798684344) q[3];
rz(3.0688799844102213) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.7131292746635004) q[0];
rz(1.54340489745349) q[0];
ry(-2.2624240341946535) q[1];
rz(1.871817186432378) q[1];
ry(-2.790128893828296) q[2];
rz(-0.1085092138812911) q[2];
ry(1.0898573967628258) q[3];
rz(-2.9630963788676934) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.9437758700562641) q[0];
rz(0.298504800322065) q[0];
ry(1.948658202387448) q[1];
rz(0.015421627846940388) q[1];
ry(-2.714067421101204) q[2];
rz(-1.6208977347996212) q[2];
ry(-2.799609995722827) q[3];
rz(-2.307684430554935) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.4224037135634715) q[0];
rz(2.778227083805787) q[0];
ry(3.139980328053253) q[1];
rz(2.1933643944402252) q[1];
ry(-2.645667834338641) q[2];
rz(-3.031798733227068) q[2];
ry(-0.9345723122229276) q[3];
rz(2.5476037056802068) q[3];