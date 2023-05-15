OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(1.5803986276235478) q[0];
rz(0.024981425037024895) q[0];
ry(0.04849753412820856) q[1];
rz(2.775936336249357) q[1];
ry(1.5705652470306115) q[2];
rz(-1.6870981514233048) q[2];
ry(-1.5725679343259227) q[3];
rz(-1.5346490973416342) q[3];
ry(-0.052252386061419465) q[4];
rz(-0.31298900683543884) q[4];
ry(-1.2231415893565571) q[5];
rz(0.727358688449219) q[5];
ry(-3.0858294756963245) q[6];
rz(-1.0921988788206738) q[6];
ry(3.056000911228022) q[7];
rz(1.417977406513371) q[7];
ry(-1.892440365820546) q[8];
rz(-2.336982245843164) q[8];
ry(-3.071222139610906) q[9];
rz(2.902141897972058) q[9];
ry(-1.530070312098316) q[10];
rz(-1.2083836126881022) q[10];
ry(-1.5838951917618553) q[11];
rz(1.9707388848646117) q[11];
ry(-3.0704732063432894) q[12];
rz(0.019800226756458095) q[12];
ry(-2.4696907167545694) q[13];
rz(3.125382147479552) q[13];
ry(2.314093536134218) q[14];
rz(-0.16303542477715274) q[14];
ry(2.4563588293378174) q[15];
rz(-0.014463531346623705) q[15];
ry(-1.5685669184292224) q[16];
rz(2.1116478015189655) q[16];
ry(1.5366006248373907) q[17];
rz(-2.901937317815527) q[17];
ry(3.093133200035) q[18];
rz(-0.19459436777454628) q[18];
ry(1.573320542487541) q[19];
rz(-0.01575466637058914) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.6454167148341545) q[0];
rz(-2.77456487833214) q[0];
ry(-1.5707473814907997) q[1];
rz(1.5762928251952435) q[1];
ry(-1.5780812362239036) q[2];
rz(-2.293991219560805) q[2];
ry(1.7748334492108158) q[3];
rz(1.2263994886483989) q[3];
ry(1.5712673225402893) q[4];
rz(-1.7784461640322098) q[4];
ry(-2.665605379074177) q[5];
rz(0.7695059215434412) q[5];
ry(-0.052293811186236744) q[6];
rz(2.960262937042365) q[6];
ry(3.085993918773601) q[7];
rz(1.0688937468278026) q[7];
ry(-0.34350476322406553) q[8];
rz(-0.8531417000200956) q[8];
ry(-1.5669702447498872) q[9];
rz(2.4030683418583827) q[9];
ry(-0.798401799389818) q[10];
rz(-1.2557895676286357) q[10];
ry(-1.752635473709491) q[11];
rz(2.2537920633916086) q[11];
ry(0.9713954204062842) q[12];
rz(1.5010450282637633) q[12];
ry(3.097769369504375) q[13];
rz(3.12526667256257) q[13];
ry(0.001989523838087592) q[14];
rz(0.1635863059598951) q[14];
ry(0.9142420083177702) q[15];
rz(3.053399330669209) q[15];
ry(3.042793295420435) q[16];
rz(2.065403282644816) q[16];
ry(-1.555320484324839) q[17];
rz(2.1559332712155412) q[17];
ry(1.5741046188852361) q[18];
rz(1.5852311568980273) q[18];
ry(-1.4856041306342052) q[19];
rz(-2.952300713630312) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.5707878065761314) q[0];
rz(1.6505854968155174) q[0];
ry(2.410352938499429) q[1];
rz(3.114368345276588) q[1];
ry(-3.0955475849243697) q[2];
rz(-2.75274964631227) q[2];
ry(3.0921594529643457) q[3];
rz(3.022521470452708) q[3];
ry(-0.046999592957194025) q[4];
rz(0.210334651678543) q[4];
ry(-1.5707277058231384) q[5];
rz(-1.5697955057734323) q[5];
ry(0.024440100804794973) q[6];
rz(2.3389427714306255) q[6];
ry(0.03173993538000186) q[7];
rz(-1.2283124016599791) q[7];
ry(-1.5820928624179647) q[8];
rz(-1.4913113247537126) q[8];
ry(3.073387819172801) q[9];
rz(-2.2140104211926532) q[9];
ry(-3.0888618449254532) q[10];
rz(-1.627749919335082) q[10];
ry(3.081249108473914) q[11];
rz(-2.4358305040448167) q[11];
ry(3.0730380774628827) q[12];
rz(-3.1153023151464225) q[12];
ry(2.157310373901728) q[13];
rz(-1.8746739319444663) q[13];
ry(2.0824613638230067) q[14];
rz(1.9162820356030237) q[14];
ry(-2.4555156586375397) q[15];
rz(-1.6557189052266315) q[15];
ry(-0.05384591305554842) q[16];
rz(-0.6138507557374161) q[16];
ry(3.094845341608723) q[17];
rz(-1.1701305959729844) q[17];
ry(-0.5857715222482983) q[18];
rz(2.719205870975574) q[18];
ry(-1.56600516627128) q[19];
rz(-1.357059221784476) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.599906078940455) q[0];
rz(-2.4117736228569298) q[0];
ry(1.648947349766775) q[1];
rz(2.2462524246068623) q[1];
ry(-3.0485872356002313) q[2];
rz(2.629738272447132) q[2];
ry(-0.008334297433869153) q[3];
rz(-2.695923043512125) q[3];
ry(1.5654455574637038) q[4];
rz(-2.4709728304369523) q[4];
ry(1.5760660438953105) q[5];
rz(0.6888429479783946) q[5];
ry(-3.1415124352978903) q[6];
rz(-2.151129940995898) q[6];
ry(3.1409099044483755) q[7];
rz(2.3132431388147516) q[7];
ry(1.553437443566584) q[8];
rz(1.221255606697605) q[8];
ry(-1.5471471147183964) q[9];
rz(-1.8423426577684137) q[9];
ry(-2.955495265041725) q[10];
rz(-0.6504368146825684) q[10];
ry(-2.5885816158435486) q[11];
rz(-0.09962050847293646) q[11];
ry(-2.1026365258057726) q[12];
rz(-3.0702361734229267) q[12];
ry(2.0202194616951945) q[13];
rz(-0.7053405147868195) q[13];
ry(1.11951460699205) q[14];
rz(2.4368264763508702) q[14];
ry(2.1313153722738543) q[15];
rz(-0.06296268588356746) q[15];
ry(-0.5736856507708998) q[16];
rz(3.107064981151379) q[16];
ry(2.7024736313492013) q[17];
rz(2.8492524297226254) q[17];
ry(1.4005904166302219) q[18];
rz(-2.2300649444070677) q[18];
ry(1.1831740356692217) q[19];
rz(2.3347732395110974) q[19];