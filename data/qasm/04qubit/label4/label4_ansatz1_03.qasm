OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.9156624087584309) q[0];
rz(-2.401223679775111) q[0];
ry(2.635595153091959) q[1];
rz(0.07214582003598659) q[1];
ry(1.1894295270742516) q[2];
rz(1.4986859981908678) q[2];
ry(-1.7570648642203106) q[3];
rz(1.5152086975708) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.9853312839320214) q[0];
rz(3.0167125411504636) q[0];
ry(-1.815402532979679) q[1];
rz(1.5625611207938281) q[1];
ry(1.9447317151267443) q[2];
rz(0.5123082723285481) q[2];
ry(1.700608481411723) q[3];
rz(0.30298241132463144) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.02236771520950409) q[0];
rz(1.4999673948725705) q[0];
ry(1.493422821689453) q[1];
rz(1.8601516331625918) q[1];
ry(-0.7095009647272212) q[2];
rz(2.248430754686195) q[2];
ry(0.00560776536109664) q[3];
rz(2.079852932146709) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.5264613134706826) q[0];
rz(1.9401999131135677) q[0];
ry(-0.6845930630932582) q[1];
rz(0.6505236247307026) q[1];
ry(-2.5127865982569837) q[2];
rz(-1.1345561995762632) q[2];
ry(2.209816304634921) q[3];
rz(-0.9075282576775452) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.517769923841569) q[0];
rz(-0.4430854772305235) q[0];
ry(-0.24801613613157336) q[1];
rz(1.2388471753583659) q[1];
ry(-0.49373848552347394) q[2];
rz(-1.935036329238394) q[2];
ry(0.6876244333650989) q[3];
rz(2.2687239456646804) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.3608629625795629) q[0];
rz(-1.857125682080981) q[0];
ry(-0.8419127891662805) q[1];
rz(-2.7303183616619635) q[1];
ry(1.8785012487868387) q[2];
rz(-1.5595997252999247) q[2];
ry(2.8482649178813784) q[3];
rz(-2.1987089533924076) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.1296586807781126) q[0];
rz(-1.2980170716317385) q[0];
ry(1.2346529348905628) q[1];
rz(-1.0511267675677338) q[1];
ry(1.310174295552264) q[2];
rz(2.1782713927463027) q[2];
ry(-1.9957119016334761) q[3];
rz(2.9948571315222523) q[3];