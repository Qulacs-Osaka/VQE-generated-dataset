OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.3694870210261005) q[0];
rz(1.0965227908600532) q[0];
ry(-1.2256445275780248) q[1];
rz(-0.9327333977517346) q[1];
ry(-0.0047424312069088625) q[2];
rz(-1.2914719455040498) q[2];
ry(0.012336233841949706) q[3];
rz(2.5128558125564155) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.0409153944172755) q[0];
rz(-0.1315908635095857) q[0];
ry(-1.1698342486630597) q[1];
rz(-2.0500419219791493) q[1];
ry(-0.0016195502030065254) q[2];
rz(-0.2957752941270821) q[2];
ry(3.0075110325649743) q[3];
rz(-2.641632716071558) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.7664931353935103) q[0];
rz(-1.2553143471039871) q[0];
ry(1.4349969540083485) q[1];
rz(1.5907682419932647) q[1];
ry(1.508862876853482) q[2];
rz(1.515143939007213) q[2];
ry(1.678128784314625) q[3];
rz(-0.6210821847898279) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.1050092340409936) q[0];
rz(0.830768670235252) q[0];
ry(-3.0368863669593154) q[1];
rz(-0.8763245658760502) q[1];
ry(1.289195410139988) q[2];
rz(-3.0582773912389682) q[2];
ry(-3.0385002544922166) q[3];
rz(-0.9911672373478514) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.945036719066156) q[0];
rz(0.024448959918006184) q[0];
ry(-1.9179358731254492) q[1];
rz(1.1002975432896474) q[1];
ry(-0.5701579517228641) q[2];
rz(1.4722907828403784) q[2];
ry(3.1307352960271877) q[3];
rz(-0.7278435024861315) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.0045254299859005505) q[0];
rz(2.3641970898648252) q[0];
ry(-3.1401050124480343) q[1];
rz(0.368978448764901) q[1];
ry(-1.4776788364106856) q[2];
rz(1.5266334773013632) q[2];
ry(3.0098603922420173) q[3];
rz(2.9581316848422854) q[3];