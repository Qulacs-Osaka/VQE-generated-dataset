OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.2485640617119333) q[0];
rz(1.7408786570160952) q[0];
ry(-1.5065980915860118) q[1];
rz(0.12413213635837028) q[1];
ry(3.123348520909314) q[2];
rz(0.2551676023050158) q[2];
ry(1.597069161639447) q[3];
rz(-2.232763717651924) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.389839211567167) q[0];
rz(-2.0107321230438093) q[0];
ry(-0.9541743543855377) q[1];
rz(-1.7645745095891445) q[1];
ry(-2.596471320622318) q[2];
rz(-2.0863951596938426) q[2];
ry(-0.10015825398939582) q[3];
rz(-1.637545820780807) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.9252276578403142) q[0];
rz(0.7229634191622302) q[0];
ry(-1.402306707195889) q[1];
rz(-1.6128383483724864) q[1];
ry(-2.0494045928197875) q[2];
rz(-0.3569117575206132) q[2];
ry(0.16183861943924863) q[3];
rz(-2.394754516292054) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.7252022542480896) q[0];
rz(-1.34320305248847) q[0];
ry(-0.12772450526259238) q[1];
rz(-1.4926284470198559) q[1];
ry(-1.8899897031641935) q[2];
rz(-2.7459659529182128) q[2];
ry(-0.4148292238518571) q[3];
rz(-2.88665381503094) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.7170282182427741) q[0];
rz(2.3664770667765245) q[0];
ry(2.9354741077544597) q[1];
rz(-2.446048718616447) q[1];
ry(-1.9259488605916837) q[2];
rz(-0.9340811034882633) q[2];
ry(0.004019712072480762) q[3];
rz(2.0066519271759686) q[3];