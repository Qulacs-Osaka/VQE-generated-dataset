OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.561519319400829) q[0];
rz(-1.7803453569672518) q[0];
ry(1.5420762471153753) q[1];
rz(1.0946906558715401) q[1];
ry(-3.1415924285372747) q[2];
rz(2.612437971154039) q[2];
ry(-3.0757097708950045) q[3];
rz(-0.35116437648057147) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.4038862742296745) q[0];
rz(-2.8970217561850893) q[0];
ry(0.3920228817889289) q[1];
rz(-0.20517770860673323) q[1];
ry(-3.1415916815301217) q[2];
rz(-0.832580890948675) q[2];
ry(-1.8472132223377793) q[3];
rz(-0.9619727263858313) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.414010125588072) q[0];
rz(0.9456787763448268) q[0];
ry(2.701302197669716) q[1];
rz(0.06779451948004707) q[1];
ry(4.1547428543939186e-07) q[2];
rz(2.8635915570107096) q[2];
ry(3.1284045083615064) q[3];
rz(-1.9037517304407565) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.5471809736885882) q[0];
rz(1.9962745135679985) q[0];
ry(-2.5087616576734604) q[1];
rz(0.403767325671965) q[1];
ry(1.5707963235785452) q[2];
rz(1.2865343736874664e-05) q[2];
ry(2.6343281941403554) q[3];
rz(2.477050069364811) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-8.793279651087005e-07) q[0];
rz(2.5707258269296696) q[0];
ry(-1.2811869769182183e-06) q[1];
rz(2.569932433904085) q[1];
ry(1.5707963915765522) q[2];
rz(2.366261051901959) q[2];
ry(3.141580137295417) q[3];
rz(-2.412793684268109) q[3];