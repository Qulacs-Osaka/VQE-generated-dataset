OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(2.1181575293166652) q[0];
rz(-1.8191598287727766) q[0];
ry(-0.47739488018967075) q[1];
rz(1.8454618388666617) q[1];
ry(-1.5639123515252367) q[2];
rz(-1.475125021962099) q[2];
ry(-0.4200755216688904) q[3];
rz(-3.109602265687799) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.479348455917214) q[0];
rz(2.633140612420019) q[0];
ry(1.5713869073252067) q[1];
rz(4.164634100031132e-06) q[1];
ry(-0.09372809854691866) q[2];
rz(3.1415305941739153) q[2];
ry(1.617094321573087) q[3];
rz(-2.6484271848936634) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.567392233132514) q[0];
rz(-0.47559290323208986) q[0];
ry(-0.3993850045040253) q[1];
rz(3.141575128048535) q[1];
ry(-3.1090763166626263) q[2];
rz(-0.00024468502873720155) q[2];
ry(0.19766091640595373) q[3];
rz(0.34178655842602623) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-3.1414276280519116) q[0];
rz(1.098282771313831) q[0];
ry(1.6457936156680324) q[1];
rz(1.5707002131774903) q[1];
ry(-0.16120191212424828) q[2];
rz(-1.5705559868671852) q[2];
ry(3.0734277561763568) q[3];
rz(0.8025327368002914) q[3];