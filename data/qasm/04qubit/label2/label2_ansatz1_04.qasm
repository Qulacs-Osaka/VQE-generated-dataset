OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.9737444378977864) q[0];
rz(0.49135589704531846) q[0];
ry(0.7109381031087603) q[1];
rz(1.64577671070299) q[1];
ry(2.61338696973302) q[2];
rz(0.21432630745227055) q[2];
ry(-1.4132274326560061) q[3];
rz(2.557165957466856) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(3.066485623712276) q[0];
rz(0.7714944319435388) q[0];
ry(2.489669981644142) q[1];
rz(-1.6214525430089415) q[1];
ry(-0.3209047625773609) q[2];
rz(-0.10769272966109815) q[2];
ry(-2.2201888056870986) q[3];
rz(-2.5742153514048076) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.357340469869014) q[0];
rz(-2.2189080441830296) q[0];
ry(-2.724232501611859) q[1];
rz(-1.720687003024242) q[1];
ry(-2.3559454255920946) q[2];
rz(-3.0742889858219886) q[2];
ry(2.601945191305196) q[3];
rz(1.4243906590742443) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(3.1032809153437073) q[0];
rz(-0.5419603221853259) q[0];
ry(-1.6203139552107968) q[1];
rz(-0.6035856131381069) q[1];
ry(1.3294340031380925) q[2];
rz(0.24733130774860257) q[2];
ry(-2.440199089107707) q[3];
rz(0.7816363633994117) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.3928423970805226) q[0];
rz(2.196267065006195) q[0];
ry(-2.9777783396612096) q[1];
rz(3.0751317735900643) q[1];
ry(-0.8579568443338133) q[2];
rz(2.98515667522648) q[2];
ry(-1.4384438271898434) q[3];
rz(-2.7732512389428914) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.7183809318529915) q[0];
rz(2.2968457469985637) q[0];
ry(-0.2708497283647331) q[1];
rz(2.6772406335347663) q[1];
ry(2.806709302581549) q[2];
rz(0.6975569415772905) q[2];
ry(-0.1109533040677837) q[3];
rz(1.9827789699514895) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.0556621353169344) q[0];
rz(-1.1527160258564837) q[0];
ry(1.526032371653642) q[1];
rz(2.007505915853102) q[1];
ry(-2.8596584310393895) q[2];
rz(0.15587536803738664) q[2];
ry(-0.167874741270467) q[3];
rz(0.5299778116065822) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.6813959179703241) q[0];
rz(-1.3638870461176518) q[0];
ry(-2.1165395796365947) q[1];
rz(-2.753408303238419) q[1];
ry(-0.3820733765771258) q[2];
rz(2.33382088794796) q[2];
ry(-1.4684873957400688) q[3];
rz(0.6344630544092169) q[3];