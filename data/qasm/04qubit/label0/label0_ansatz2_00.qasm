OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(3.1415761329149015) q[0];
rz(2.2117625361108484) q[0];
ry(-1.6349955334123172) q[1];
rz(-1.9746240917040865) q[1];
ry(-1.5804442625872452) q[2];
rz(-3.0010680075162997) q[2];
ry(2.882144888481065) q[3];
rz(-2.7892593097384446) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-3.1415810954282612) q[0];
rz(-0.5501804471342682) q[0];
ry(1.4452989370466138) q[1];
rz(-2.21477980714102) q[1];
ry(0.9144792458186091) q[2];
rz(1.3332427180639685) q[2];
ry(-0.007367168488723763) q[3];
rz(3.0375277509390397) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.5698820559443671) q[0];
rz(-1.5705731626269008) q[0];
ry(1.5515862051251004) q[1];
rz(-0.12602507149762673) q[1];
ry(-1.5244402147302312) q[2];
rz(-0.06297998129201007) q[2];
ry(0.02672196133884963) q[3];
rz(-1.892438714892055) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.3402836755915715) q[0];
rz(1.5698248553042853) q[0];
ry(1.5707420478812208) q[1];
rz(3.1413677564063014) q[1];
ry(1.5708125788379723) q[2];
rz(-3.1415492479196634) q[2];
ry(1.5708022686609497) q[3];
rz(-6.180922726159229e-06) q[3];