OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.5996202798388097) q[0];
rz(1.286932305595652) q[0];
ry(0.33955170819550295) q[1];
rz(-1.769057692418606) q[1];
ry(2.9032449997012284) q[2];
rz(-2.218441828032689) q[2];
ry(-1.160235532514827) q[3];
rz(-1.254167035823187) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.28892795983107256) q[0];
rz(2.8297259366450613) q[0];
ry(0.15439163583814786) q[1];
rz(1.5480819503177154) q[1];
ry(2.1767136728316405) q[2];
rz(2.020108376795358) q[2];
ry(-0.6887863767821081) q[3];
rz(2.5883164400184873) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.2994576285596873) q[0];
rz(-2.2010036896435983) q[0];
ry(0.14964795028974442) q[1];
rz(-2.573806140593113) q[1];
ry(-2.915997211040834) q[2];
rz(-0.8132803869232864) q[2];
ry(1.3048988033963393) q[3];
rz(0.9940026631507897) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.3603435121005541) q[0];
rz(1.7680570972978682) q[0];
ry(0.9969833251177153) q[1];
rz(-2.7827822991288285) q[1];
ry(-0.13418060367604756) q[2];
rz(-1.6768234250111227) q[2];
ry(-2.642757102154987) q[3];
rz(0.035368705674190266) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.8009674651558498) q[0];
rz(2.7695463258102713) q[0];
ry(3.010150485056423) q[1];
rz(-3.1081450584522403) q[1];
ry(-0.36780581675275287) q[2];
rz(-2.981997251617314) q[2];
ry(1.906014693990942) q[3];
rz(2.306632335197872) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.8964354439844486) q[0];
rz(2.975310317560833) q[0];
ry(-2.1117063884507554) q[1];
rz(-1.2628515844672332) q[1];
ry(1.017483149224163) q[2];
rz(-2.9480105406023087) q[2];
ry(1.3442409793239047) q[3];
rz(1.2376525646568395) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.962552148653274) q[0];
rz(0.6090903604774962) q[0];
ry(3.1274068468139546) q[1];
rz(-1.8335988801409338) q[1];
ry(-0.20500065621559407) q[2];
rz(2.8731186424314528) q[2];
ry(-1.2307589975969206) q[3];
rz(0.1290082782939977) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.2332201918304665) q[0];
rz(-2.579313741792353) q[0];
ry(2.1234104057771113) q[1];
rz(1.3306834496722428) q[1];
ry(-0.6127385289379763) q[2];
rz(2.813081892996353) q[2];
ry(0.4255665165053218) q[3];
rz(0.33228151193535277) q[3];