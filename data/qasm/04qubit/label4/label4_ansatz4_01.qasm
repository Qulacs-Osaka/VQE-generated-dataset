OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.5009331404404724) q[0];
rz(2.846523840531543) q[0];
ry(2.105823595453688) q[1];
rz(-1.2156685767988824) q[1];
ry(-1.6503206877969123) q[2];
rz(2.437425514642834) q[2];
ry(0.1491729369116923) q[3];
rz(-3.1056546701788315) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.34361438345705e-05) q[0];
rz(1.8063970221665278) q[0];
ry(-4.082273554824001e-06) q[1];
rz(2.4600075981737466) q[1];
ry(3.198775246104901e-05) q[2];
rz(0.7041645898635902) q[2];
ry(4.534285360093744e-06) q[3];
rz(-0.7533665497666603) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.866747986943044) q[0];
rz(0.2659873937708497) q[0];
ry(-1.0748253390791627) q[1];
rz(2.189176653742881) q[1];
ry(1.4912735272559743) q[2];
rz(-0.1942367033238126) q[2];
ry(1.683558787267606) q[3];
rz(3.0437411040292166) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.7438469134365187) q[0];
rz(2.367312131750064) q[0];
ry(-1.8543343271720012) q[1];
rz(-3.141592123332546) q[1];
ry(1.570790044323524) q[2];
rz(0.08664447004994535) q[2];
ry(-2.0686235103615065) q[3];
rz(-3.141565186827133) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.5707895106512302) q[0];
rz(1.3691087861012292) q[0];
ry(-1.5707969657267478) q[1];
rz(0.2725784762022432) q[1];
ry(3.1415899678794252) q[2];
rz(-0.11504136840163334) q[2];
ry(1.5707889921340876) q[3];
rz(1.8433679813059036) q[3];