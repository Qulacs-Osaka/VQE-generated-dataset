OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.4740759618760566) q[0];
rz(2.058489598034635) q[0];
ry(4.16837519491678e-06) q[1];
rz(0.5046659593013496) q[1];
ry(7.986148907512813e-06) q[2];
rz(0.4617584565181021) q[2];
ry(0.40055514674678516) q[3];
rz(-2.072126202697994) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.9374259558299127) q[0];
rz(2.067228768689282) q[0];
ry(-1.9788440384184236e-06) q[1];
rz(-0.5220156978348561) q[1];
ry(1.5707951386517776) q[2];
rz(0.11052635746189547) q[2];
ry(1.7715614379671483) q[3];
rz(1.9197851276673301) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.5707902390669308) q[0];
rz(-3.0828008345490248) q[0];
ry(-1.5707990767356654) q[1];
rz(0.8928694909128051) q[1];
ry(-2.443790805273238e-07) q[2];
rz(-0.9495408506825472) q[2];
ry(2.7905294393363866) q[3];
rz(-1.5707953853028913) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.1415915415396722) q[0];
rz(0.46672580714129985) q[0];
ry(-4.366718728751892e-07) q[1];
rz(0.14530509999227004) q[1];
ry(-3.3351401125258917e-06) q[2];
rz(-1.5435817283165973) q[2];
ry(1.5707981950720171) q[3];
rz(-2.1034185014935494) q[3];