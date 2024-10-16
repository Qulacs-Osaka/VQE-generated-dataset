OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(3.141519730466529) q[0];
rz(-2.0707847863465827) q[0];
ry(-3.1415924274787907) q[1];
rz(-1.2805685863307203) q[1];
ry(3.1415735873394772) q[2];
rz(-2.3631944950472006) q[2];
ry(-3.1415847379344695) q[3];
rz(-2.2725578088879317) q[3];
ry(-3.1395317361015955) q[4];
rz(0.1250418874577779) q[4];
ry(-3.1415919216782973) q[5];
rz(-2.8118420695335886) q[5];
ry(3.1323148418532787) q[6];
rz(2.2203798006074873) q[6];
ry(1.6809692917618688e-06) q[7];
rz(0.22874922767806072) q[7];
ry(1.9750127801927222) q[8];
rz(-1.8561129509141905) q[8];
ry(-3.1415867794797436) q[9];
rz(2.804052543989167) q[9];
ry(-1.5717948421572883) q[10];
rz(-3.1415631401670514) q[10];
ry(0.8452376868173923) q[11];
rz(1.589252889730525) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.15727697795448492) q[0];
rz(-1.4440536033895217) q[0];
ry(1.5707969430598474) q[1];
rz(-1.640458192241142) q[1];
ry(1.2060056792334706) q[2];
rz(1.7608843077258145) q[2];
ry(-1.5752290752679574) q[3];
rz(0.0005786359751769728) q[3];
ry(3.1115293950334313) q[4];
rz(-0.7575509314300071) q[4];
ry(1.1425748211074892e-06) q[5];
rz(1.4146464570236805) q[5];
ry(7.489493523316072e-05) q[6];
rz(-0.3452743271581387) q[6];
ry(3.3405466925984234e-07) q[7];
rz(-0.9053615111708685) q[7];
ry(6.673652685261478e-05) q[8];
rz(0.09449991127555625) q[8];
ry(1.993625371454755e-07) q[9];
rz(0.5814363987120394) q[9];
ry(-1.5705886451358766) q[10];
rz(-0.460565106617862) q[10];
ry(-3.679563271700499e-05) q[11];
rz(-1.5890755738189124) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-3.141591975916329) q[0];
rz(1.6975796603562614) q[0];
ry(6.845858025794017e-05) q[1];
rz(1.6404540640794694) q[1];
ry(-3.545261870741001e-07) q[2];
rz(1.380623496235203) q[2];
ry(-1.5707992229001713) q[3];
rz(3.141589845098727) q[3];
ry(-3.1415914091560366) q[4];
rz(2.382524723761572) q[4];
ry(-3.1347971423328507) q[5];
rz(-1.8187366711317248) q[5];
ry(2.5359461446328173e-06) q[6];
rz(1.2920276029708184) q[6];
ry(-1.6587074971009903) q[7];
rz(1.7805972970403356) q[7];
ry(1.6632614515014342e-06) q[8];
rz(-1.3803115953151988) q[8];
ry(0.15126078713827965) q[9];
rz(0.34438444485128955) q[9];
ry(-2.458958037188097e-06) q[10];
rz(0.46113802604916904) q[10];
ry(1.5707997422179965) q[11];
rz(3.1415899417997784) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.5651877679381827) q[0];
rz(-2.6548002608172374) q[0];
ry(1.5707953267832018) q[1];
rz(-1.0047505398356462) q[1];
ry(-3.012679726603087) q[2];
rz(-2.654887020094967) q[2];
ry(-1.5707956485833074) q[3];
rz(-1.0047731226665866) q[3];
ry(2.38474545765376) q[4];
rz(0.4867775777401895) q[4];
ry(3.1415900877242877) q[5];
rz(-0.08705854365552276) q[5];
ry(1.594427460232148) q[6];
rz(-2.6548002011845817) q[6];
ry(-8.24109019607582e-07) q[7];
rz(-3.0629259061483096) q[7];
ry(-2.927125406060342) q[8];
rz(0.4865196148365563) q[8];
ry(-3.7122941343170623e-07) q[9];
rz(1.5150970366237833) q[9];
ry(0.24255525783003762) q[10];
rz(0.4862853137618855) q[10];
ry(1.5707978968799372) q[11];
rz(-1.2823623753069824) q[11];