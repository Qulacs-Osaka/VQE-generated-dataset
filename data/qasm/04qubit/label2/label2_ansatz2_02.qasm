OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.7371575694763974) q[0];
rz(1.7548181912047935) q[0];
ry(0.03287555645265496) q[1];
rz(1.5658472489222963) q[1];
ry(-1.5120394134972805) q[2];
rz(3.135513108987021) q[2];
ry(1.5975705930505175) q[3];
rz(0.15902846973937004) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.9810723249066224) q[0];
rz(-0.818856962445434) q[0];
ry(0.6366526751276673) q[1];
rz(1.6089301609336377) q[1];
ry(-1.4686840189028487) q[2];
rz(1.2093009745105263) q[2];
ry(-3.129756671335655) q[3];
rz(2.850977137337196) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.7216838868846578) q[0];
rz(-2.4397253599682527) q[0];
ry(0.7535900133542229) q[1];
rz(-0.3160564649252872) q[1];
ry(-1.4546889114094812) q[2];
rz(-2.0238395078539058) q[2];
ry(-0.47904867341242063) q[3];
rz(0.8203441402389805) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.940151344620045) q[0];
rz(-0.6144654722345129) q[0];
ry(0.3514237776029585) q[1];
rz(-1.7805921096077226) q[1];
ry(-0.9825835827591775) q[2];
rz(-2.705472416362461) q[2];
ry(-2.5837435430213103) q[3];
rz(0.574543824650629) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-3.0852921065403685) q[0];
rz(-2.88446000341249) q[0];
ry(-1.1547495416046227) q[1];
rz(0.9415040571549096) q[1];
ry(0.985788553436234) q[2];
rz(-0.6163723292457954) q[2];
ry(-1.8525157528314875) q[3];
rz(-1.744898077901355) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.2899375503517416) q[0];
rz(-0.8840413712162938) q[0];
ry(-2.867852314597754) q[1];
rz(0.7954113527280899) q[1];
ry(1.8644559550112785) q[2];
rz(2.7299595527543263) q[2];
ry(-1.7269920469930566) q[3];
rz(-0.45335929311420486) q[3];