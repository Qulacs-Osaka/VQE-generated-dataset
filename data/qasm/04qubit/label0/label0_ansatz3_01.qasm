OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.9122301340002972) q[0];
rz(-1.3655159856673977) q[0];
ry(1.4783443523786586) q[1];
rz(-2.691784815173027) q[1];
ry(2.1777316218920078) q[2];
rz(-0.2970442027142056) q[2];
ry(-1.9364668896544863) q[3];
rz(-1.4627363747352016) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.20854095123612115) q[0];
rz(-2.8632404046365516) q[0];
ry(0.4724026597420643) q[1];
rz(-1.8409145139037753) q[1];
ry(2.16120781677768) q[2];
rz(0.5183520317289272) q[2];
ry(-0.5872790054542563) q[3];
rz(2.682851863499011) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.805982173134135) q[0];
rz(-1.6209048224191287) q[0];
ry(-0.46217054731368107) q[1];
rz(1.936257517433378) q[1];
ry(1.5817317955637866) q[2];
rz(1.5866414526092143) q[2];
ry(-0.5549118211903874) q[3];
rz(0.6562744280051653) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.043265639747576) q[0];
rz(1.5783758082130461) q[0];
ry(-0.7404030781795454) q[1];
rz(-3.117321016260808) q[1];
ry(2.0454367903749664) q[2];
rz(-1.177531346480846) q[2];
ry(0.34790937740355604) q[3];
rz(2.383737107283175) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.7469276823906207) q[0];
rz(0.9953498513847813) q[0];
ry(0.3215450879732087) q[1];
rz(-1.1382103582007008) q[1];
ry(1.979397667919511) q[2];
rz(-1.2787719973736196) q[2];
ry(2.7833379886210543) q[3];
rz(1.071017309783505) q[3];