OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(2.8047827440484525) q[0];
rz(0.24165860485162013) q[0];
ry(-2.870912010790953) q[1];
rz(-0.09689805234419013) q[1];
ry(1.1147343226269868e-07) q[2];
rz(-1.5130683025943428) q[2];
ry(-0.33957548544258115) q[3];
rz(1.598415433911938) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.822848034762039) q[0];
rz(1.9576868646229384) q[0];
ry(1.8189646304744393) q[1];
rz(2.729680672993502) q[1];
ry(-3.1415926088386112) q[2];
rz(-3.1110236221583087) q[2];
ry(1.4831671413540135) q[3];
rz(0.39468722692645203) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.5934694514071372) q[0];
rz(1.5950577927116867) q[0];
ry(2.926533357896) q[1];
rz(1.8408209535860838) q[1];
ry(1.5707964981829603) q[2];
rz(4.390274597484545e-07) q[2];
ry(-1.359110260953464) q[3];
rz(1.00771572938649) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(3.141592089007796) q[0];
rz(0.7501442963032525) q[0];
ry(-3.1415914677266508) q[1];
rz(-0.8894416093241482) q[1];
ry(1.5707965150116576) q[2];
rz(-0.8307005776516154) q[2];
ry(-1.1992622089769905e-06) q[3];
rz(2.1542704267283197) q[3];