OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.0008230110588041) q[0];
rz(-0.6825685980098645) q[0];
ry(-2.306919371120745) q[1];
rz(-1.6745166351337875) q[1];
ry(-1.5087229113439682) q[2];
rz(0.9201800244535185) q[2];
ry(3.001108608008715) q[3];
rz(2.0840150944844438) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.3742226929965025) q[0];
rz(-3.0337943574699993) q[0];
ry(2.3289352050854433) q[1];
rz(0.5465889487651241) q[1];
ry(-1.1801535719481007) q[2];
rz(-2.7311508601114043) q[2];
ry(-2.1443995280561374) q[3];
rz(-1.342498249976762) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.257509757172306) q[0];
rz(2.4523237531634825) q[0];
ry(1.640020310435549) q[1];
rz(0.6097443461566961) q[1];
ry(-1.4445938964583362) q[2];
rz(2.718620890898862) q[2];
ry(-1.1620575089957934) q[3];
rz(-2.7572687604167756) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.9661861378473315) q[0];
rz(-2.6197265172481883) q[0];
ry(2.9766254948937223) q[1];
rz(0.7174300151489775) q[1];
ry(0.7524925982121058) q[2];
rz(2.2121238720039496) q[2];
ry(-0.7481628588359232) q[3];
rz(0.7088856120178084) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.2387741331878237) q[0];
rz(-0.9906060310451377) q[0];
ry(0.6923560191041558) q[1];
rz(1.7860646023955153) q[1];
ry(-1.357766992882909) q[2];
rz(0.8058380653585697) q[2];
ry(-1.417965776814067) q[3];
rz(-2.009870597894853) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.935858768652918) q[0];
rz(2.887825073900005) q[0];
ry(-0.7132803261799157) q[1];
rz(-2.212030135072216) q[1];
ry(-0.12005886843332694) q[2];
rz(-0.11411693084062109) q[2];
ry(-1.7060182252723164) q[3];
rz(2.3811504261204193) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.9520920381060135) q[0];
rz(-2.4877995969313784) q[0];
ry(0.5443621795195447) q[1];
rz(0.16952498678200534) q[1];
ry(-2.4789536438985458) q[2];
rz(1.5530980558092526) q[2];
ry(0.15745340042134764) q[3];
rz(-1.2395270400742135) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.092333755760257) q[0];
rz(-2.4202661744555627) q[0];
ry(0.6367193986213051) q[1];
rz(0.5669282131276105) q[1];
ry(-2.4829780271374258) q[2];
rz(2.231726293779769) q[2];
ry(2.631117683382646) q[3];
rz(2.2894189897979245) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.13929589391261565) q[0];
rz(1.375345553908934) q[0];
ry(0.7786951999632397) q[1];
rz(-1.6329187970511896) q[1];
ry(-0.8467947402087344) q[2];
rz(-1.7228465884428887) q[2];
ry(-2.5099509389022305) q[3];
rz(-1.4311288867176775) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.4310786799963076) q[0];
rz(-1.4742872282419623) q[0];
ry(-1.5902174611422313) q[1];
rz(2.045793252655967) q[1];
ry(-0.13641256438296256) q[2];
rz(-1.4506821622549508) q[2];
ry(-1.390585576770007) q[3];
rz(2.067292000232559) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.2015929201654245) q[0];
rz(1.783933919528673) q[0];
ry(-0.914534954563841) q[1];
rz(-0.03152246794454694) q[1];
ry(-3.0342176585201406) q[2];
rz(-1.6415613076854116) q[2];
ry(-2.3658974187718593) q[3];
rz(-2.68516055037182) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.905078438616272) q[0];
rz(-0.2534532489755769) q[0];
ry(1.056638615846114) q[1];
rz(-0.014290181703464631) q[1];
ry(1.5949399162699658) q[2];
rz(1.130810963941287) q[2];
ry(0.5859093876856023) q[3];
rz(2.2170820141622842) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.9371816282645401) q[0];
rz(1.5730264064672452) q[0];
ry(-2.59856983339694) q[1];
rz(2.905362674231527) q[1];
ry(-1.4838044559334382) q[2];
rz(2.6651668079960382) q[2];
ry(-1.7319762215165697) q[3];
rz(-0.35121124035373086) q[3];