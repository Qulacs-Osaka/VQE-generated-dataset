OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.8213597265742267) q[0];
rz(0.4053484584105362) q[0];
ry(-0.40536080035503286) q[1];
rz(1.3946807901026526) q[1];
ry(0.9639076717240531) q[2];
rz(1.2010455548197996) q[2];
ry(0.4270344029887695) q[3];
rz(-0.6854584649505452) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.928274640594738) q[0];
rz(-2.9095187753651213) q[0];
ry(-0.8740078622982095) q[1];
rz(-1.62134075359824) q[1];
ry(-0.9762545972075104) q[2];
rz(0.8036718511196433) q[2];
ry(-1.5360051669464951) q[3];
rz(2.6050352502513667) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.893357309216194) q[0];
rz(-0.41686329207329964) q[0];
ry(1.5001967854886802) q[1];
rz(0.8180347716723366) q[1];
ry(-1.4045969383749304) q[2];
rz(-1.0475027413947977) q[2];
ry(2.962400500169285) q[3];
rz(2.9247199317482266) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.9605484661401218) q[0];
rz(1.3747981571403907) q[0];
ry(-1.156695767061357) q[1];
rz(2.7344633557662075) q[1];
ry(-0.20433791953028696) q[2];
rz(-0.9441816222624925) q[2];
ry(-2.4235442217679117) q[3];
rz(-1.1070268609238525) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.2816276964220998) q[0];
rz(2.0798132893455983) q[0];
ry(2.1491424606019964) q[1];
rz(0.3225441416722825) q[1];
ry(-2.350332706661099) q[2];
rz(2.3988746000923595) q[2];
ry(-0.31729204901800084) q[3];
rz(2.212994850848726) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.7545994102557345) q[0];
rz(-1.7258333848168783) q[0];
ry(1.2305188822256938) q[1];
rz(-1.0404897554158712) q[1];
ry(1.2833653057277736) q[2];
rz(-2.6806602421836696) q[2];
ry(0.49222990974839964) q[3];
rz(2.264457221095821) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(3.114132611064258) q[0];
rz(-0.6535756802943692) q[0];
ry(0.4669669861036622) q[1];
rz(-1.728678127737708) q[1];
ry(-1.80088065105056) q[2];
rz(2.729362373439992) q[2];
ry(-0.923134889094447) q[3];
rz(2.7562965699437862) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.2543182519169505) q[0];
rz(1.134215982909227) q[0];
ry(-2.2600109945508193) q[1];
rz(1.1240984738562911) q[1];
ry(2.9454289079526417) q[2];
rz(-1.344032685333162) q[2];
ry(2.6629558942788067) q[3];
rz(2.2892675384418397) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.7997211209983575) q[0];
rz(0.7315507914480691) q[0];
ry(-0.6349401023474553) q[1];
rz(-3.020272975764704) q[1];
ry(-1.019500290674599) q[2];
rz(1.9433165921960693) q[2];
ry(0.6271751996695549) q[3];
rz(0.0038703195750754804) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.5023482215288668) q[0];
rz(0.1411983945812471) q[0];
ry(0.27073725061488485) q[1];
rz(0.2501022500896156) q[1];
ry(-2.6697031243950606) q[2];
rz(2.9217546194642336) q[2];
ry(1.1258107313231394) q[3];
rz(0.8106681042241427) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.16797543322797726) q[0];
rz(1.7251971734898168) q[0];
ry(0.4573812735460736) q[1];
rz(1.4499692596262719) q[1];
ry(1.9356338707117997) q[2];
rz(0.5645916522072174) q[2];
ry(-0.5790561134844046) q[3];
rz(1.4439041850655139) q[3];