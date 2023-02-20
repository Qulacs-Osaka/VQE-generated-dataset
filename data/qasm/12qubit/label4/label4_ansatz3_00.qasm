OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.5707981489083718) q[0];
rz(-1.5708193744797985) q[0];
ry(-1.570745626254839) q[1];
rz(1.5857907294402065) q[1];
ry(3.1415919668454713) q[2];
rz(-2.117706658216969) q[2];
ry(-0.3126954872182779) q[3];
rz(1.5708319038738097) q[3];
ry(1.570799588572253) q[4];
rz(1.4799496895750315) q[4];
ry(1.570797216363227) q[5];
rz(0.4460118394317087) q[5];
ry(-1.570797187852939) q[6];
rz(-1.570958145205398) q[6];
ry(-3.141589477587978) q[7];
rz(1.0533168373106383) q[7];
ry(3.141580172697708) q[8];
rz(-2.1928093163352473) q[8];
ry(-3.1415767438852433) q[9];
rz(2.787077435167841) q[9];
ry(-2.7841424449259824) q[10];
rz(2.158420637373169) q[10];
ry(3.1415871246492593) q[11];
rz(-1.3719138072155541) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.862223853236765) q[0];
rz(1.5707943626570025) q[0];
ry(3.141570259977969) q[1];
rz(-2.031529922382876) q[1];
ry(1.5707951683212222) q[2];
rz(2.712914581295725) q[2];
ry(1.5708044396469665) q[3];
rz(1.5709535468157205) q[3];
ry(1.1157289097865642e-06) q[4];
rz(-2.9642030059364424) q[4];
ry(0.005656852094690024) q[5];
rz(2.6956191820686786) q[5];
ry(-0.09894668403864142) q[6];
rz(-3.1414301864856533) q[6];
ry(-2.7251413795292945) q[7];
rz(1.0531293614983341) q[7];
ry(-1.5707965858294335) q[8];
rz(-3.0932827924295383) q[8];
ry(3.1415717984499882) q[9];
rz(-0.9262305103105728) q[9];
ry(1.7749641476414324) q[10];
rz(-0.2954783955304284) q[10];
ry(-3.1415882514348574) q[11];
rz(1.013591329032879) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.5707944511043064) q[0];
rz(1.4668988628316868) q[0];
ry(-1.5706625803354664) q[1];
rz(-1.570658188135338) q[1];
ry(-3.141592267170514) q[2];
rz(-0.046214523268614194) q[2];
ry(0.031867581924901245) q[3];
rz(-1.9086930686220314) q[3];
ry(-3.1075768850763663) q[4];
rz(-0.0935298232856443) q[4];
ry(-2.8783294624263465) q[5];
rz(-3.130242413618312) q[5];
ry(1.5707906127241547) q[6];
rz(-1.5707864315323565) q[6];
ry(3.1856759230564076e-06) q[7];
rz(-1.0531509215723425) q[7];
ry(-3.1415883697537828) q[8];
rz(2.5852288722014207) q[8];
ry(1.5707948365877047) q[9];
rz(1.6611233231573257) q[9];
ry(1.5707985964605982) q[10];
rz(0.2952849422400039) q[10];
ry(0.7671509004912168) q[11];
rz(-1.5707853417083153) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-3.141583196418819) q[0];
rz(-0.23760767803117488) q[0];
ry(-3.0272428815298786) q[1];
rz(1.3659222421681991) q[1];
ry(-3.1415893329269085) q[2];
rz(1.7855368685633557) q[2];
ry(6.303940283913789e-05) q[3];
rz(0.13266620911111102) q[3];
ry(-1.5864768165840815e-06) q[4];
rz(-1.187284150611677) q[4];
ry(3.0171566977903765) q[5];
rz(1.3772832400553436) q[5];
ry(1.5707948277795896) q[6];
rz(0.3033930617905085) q[6];
ry(-1.5707912597407143) q[7];
rz(-1.7757105161523734) q[7];
ry(-2.0847387670641295e-06) q[8];
rz(1.363153686644373) q[8];
ry(-1.5707980764912888) q[9];
rz(-0.20491352448666927) q[9];
ry(-7.693273704312276e-06) q[10];
rz(-2.6784048904184576) q[10];
ry(1.5708009260578897) q[11];
rz(-1.775711483865326) q[11];