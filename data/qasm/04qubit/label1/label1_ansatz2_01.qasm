OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.2803186358146301) q[0];
rz(-1.0507250320150971) q[0];
ry(1.6835284407118687) q[1];
rz(-1.3881951852504262) q[1];
ry(3.6249813568645095e-06) q[2];
rz(-2.414274877785559) q[2];
ry(-0.3194342628731537) q[3];
rz(1.392428549399046) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.22543363143409678) q[0];
rz(-0.18452092431459907) q[0];
ry(0.11312553754390553) q[1];
rz(-2.9896436664931247) q[1];
ry(-3.1415872610916318) q[2];
rz(-2.8422962581251516) q[2];
ry(0.36875290730289045) q[3];
rz(0.6276557839839051) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.21672443777439232) q[0];
rz(1.3860270315019954) q[0];
ry(0.10167170337785271) q[1];
rz(1.4699189742579675) q[1];
ry(-1.2901245699303841e-05) q[2];
rz(-0.3166656441434661) q[2];
ry(0.33959806473465814) q[3];
rz(0.8962519251299711) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.9320002923877917) q[0];
rz(1.5819562385572692) q[0];
ry(-3.1328777643976533) q[1];
rz(-1.8792619586007115) q[1];
ry(-1.5707915332546447) q[2];
rz(1.5941440754119185e-06) q[2];
ry(2.9911818562185846) q[3];
rz(0.9402778880824592) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-7.498662411009605e-07) q[0];
rz(0.564598673718999) q[0];
ry(3.141592318198024) q[1];
rz(-0.5083315771321499) q[1];
ry(-1.570806832803835) q[2];
rz(-1.6127564675691266) q[2];
ry(5.859082378704355e-07) q[3];
rz(-0.691203622669939) q[3];