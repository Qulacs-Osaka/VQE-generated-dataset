OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(2.8956460406612656) q[0];
rz(1.4145304002381074) q[0];
ry(-1.659608178539906) q[1];
rz(1.6049167004377862) q[1];
ry(-0.09522996868369443) q[2];
rz(-2.7643367613217946) q[2];
ry(0.18398860349081847) q[3];
rz(-2.73187710091842) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.8017656604030208e-05) q[0];
rz(-1.904045031262716) q[0];
ry(-0.0006319083592278983) q[1];
rz(0.2305989607243788) q[1];
ry(-1.464702913202114) q[2];
rz(1.6080074477538684) q[2];
ry(-1.688947925744591) q[3];
rz(-0.0955621570781773) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.3527036930416285) q[0];
rz(-0.865981394562582) q[0];
ry(1.5200924294957057) q[1];
rz(2.537457916614287) q[1];
ry(1.4408724086709075) q[2];
rz(1.2226981592108768) q[2];
ry(-1.1386806802962977) q[3];
rz(1.6228021479916328) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.6180086232356946e-05) q[0];
rz(-2.161013067213123) q[0];
ry(1.5708019936255688) q[1];
rz(-1.5707993976996883) q[1];
ry(0.655987358673741) q[2];
rz(2.424307466164992) q[2];
ry(-1.2822799999092798) q[3];
rz(0.34582809067894793) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.4107240564019978) q[0];
rz(-1.5707942610380625) q[0];
ry(1.570796278254698) q[1];
rz(-1.5742057099790656) q[1];
ry(3.141586906514881) q[2];
rz(-2.28021173315551) q[2];
ry(1.3561410007007133) q[3];
rz(-3.0192751010711008) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.5708018675308077) q[0];
rz(-1.710492996108032) q[0];
ry(-1.3106139976878222) q[1];
rz(2.3285246005349753) q[1];
ry(-9.543120083613077e-06) q[2];
rz(-2.1299253436371925) q[2];
ry(-3.007673099902172) q[3];
rz(-0.3345005701537433) q[3];