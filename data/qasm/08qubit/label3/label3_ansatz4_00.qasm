OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(3.066735209067943) q[0];
rz(-1.570800650125938) q[0];
ry(-1.6609904758693772) q[1];
rz(2.889632249036822e-07) q[1];
ry(3.1415921866460974) q[2];
rz(0.9063033828630701) q[2];
ry(1.5707966782781897) q[3];
rz(-0.7032663304935741) q[3];
ry(3.1415926478435874) q[4];
rz(0.8024914692807242) q[4];
ry(0.3275085181175928) q[5];
rz(1.5708052018348506) q[5];
ry(3.0900535604149177) q[6];
rz(3.1415853768962427) q[6];
ry(1.2559333428404518) q[7];
rz(-0.00025627738013871826) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.5707962933771076) q[0];
rz(-3.1415924669210553) q[0];
ry(1.5707967980607522) q[1];
rz(1.5707033664652146) q[1];
ry(1.1079955890879736) q[2];
rz(-0.0009982899797087441) q[2];
ry(-2.5824493032899904e-07) q[3];
rz(0.7032626263339987) q[3];
ry(-0.11703298614395319) q[4];
rz(-1.5708220922050578) q[4];
ry(-1.5707965890366562) q[5];
rz(2.2403154906823652) q[5];
ry(-3.0617984917261953) q[6];
rz(1.9421347420197372) q[6];
ry(-1.491755311898407) q[7];
rz(1.570807661112914) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5707973876104793) q[0];
rz(0.15797711617031318) q[0];
ry(3.1311270837796417) q[1];
rz(3.141494971141878) q[1];
ry(3.140695927971866) q[2];
rz(-2.2925555912827775) q[2];
ry(-2.815312476116252) q[3];
rz(1.5707969728714177) q[3];
ry(0.006069787025142048) q[4];
rz(-2.9695496056131354) q[4];
ry(2.2540030908402287e-07) q[5];
rz(2.4720847128143433) q[5];
ry(1.3064000370377468e-07) q[6];
rz(-1.389461549770612) q[6];
ry(-1.5707981352869282) q[7];
rz(1.5707971091198332) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(3.1415892433679327) q[0];
rz(0.6260551237041735) q[0];
ry(-1.5708067173350893) q[1];
rz(-1.1027175676544843) q[1];
ry(3.1415894564807383) q[2];
rz(1.3181134019210816) q[2];
ry(1.570785956279319) q[3];
rz(-1.102717534763516) q[3];
ry(3.141592332748287) q[4];
rz(0.6401006691857933) q[4];
ry(1.5707967894787895) q[5];
rz(-2.6735124350975084) q[5];
ry(3.1415926174635684) q[6];
rz(2.59155141371595) q[6];
ry(-1.570807004636163) q[7];
rz(-1.102462811193516) q[7];