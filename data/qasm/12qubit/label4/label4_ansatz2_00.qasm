OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.9751573542830694) q[0];
rz(2.991468881734928) q[0];
ry(-1.570796423458479) q[1];
rz(-1.2874999694592768) q[1];
ry(1.5707972044172953) q[2];
rz(3.1415923929938714) q[2];
ry(1.5707967602893467) q[3];
rz(2.938506178029143) q[3];
ry(3.1415910227336052) q[4];
rz(-1.4146185116089196) q[4];
ry(6.403853770003464e-07) q[5];
rz(1.8292700822637933) q[5];
ry(1.0813333976158622e-06) q[6];
rz(-2.273891046174529) q[6];
ry(-7.033787623392411e-07) q[7];
rz(2.881889698543395) q[7];
ry(-1.574316702633125e-06) q[8];
rz(-1.2787843289676404) q[8];
ry(-3.141590689723176) q[9];
rz(1.8483523786952487) q[9];
ry(3.1415922594732786) q[10];
rz(2.8218259933348313) q[10];
ry(-3.503941137239508e-09) q[11];
rz(1.5483175592118623) q[11];
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
ry(1.2242146670013235e-06) q[0];
rz(0.1501119232728483) q[0];
ry(3.141592455425879) q[1];
rz(-1.2875002452146682) q[1];
ry(1.5705777392473166) q[2];
rz(-1.7028097230993573) q[2];
ry(-3.1413747254815236) q[3];
rz(-0.2030874683797107) q[3];
ry(3.578458338537871e-07) q[4];
rz(-2.813426482312373) q[4];
ry(-3.1415906472138606) q[5];
rz(2.4477547095464653) q[5];
ry(-1.5707947179868214) q[6];
rz(-1.5707965658786074) q[6];
ry(1.5708052497948937) q[7];
rz(-0.12759655978196527) q[7];
ry(1.5707959746643194) q[8];
rz(1.5707969750767712) q[8];
ry(-0.128503121488811) q[9];
rz(1.571060807758757) q[9];
ry(-1.7900629157673198e-10) q[10];
rz(2.2786871290933326) q[10];
ry(-7.239931146047351e-07) q[11];
rz(1.411727850413687) q[11];
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
ry(1.5707936476667026) q[0];
rz(-3.1399009067126085) q[0];
ry(1.5707960894767745) q[1];
rz(-2.8325833733778736) q[1];
ry(-1.881881809419192) q[2];
rz(1.6508028615217478) q[2];
ry(-1.5707965751136355) q[3];
rz(-3.0644979780652646) q[3];
ry(-1.5707937075897291) q[4];
rz(3.141555568558319) q[4];
ry(1.570798410455312) q[5];
rz(-0.0001524050047470027) q[5];
ry(1.5707961307979332) q[6];
rz(3.0078475694324385) q[6];
ry(-1.7676267328137023) q[7];
rz(1.6208354222643724) q[7];
ry(-1.5707977400695254) q[8];
rz(0.9993389076431435) q[8];
ry(-1.570794564522678) q[9];
rz(-0.01116263869606606) q[9];
ry(-1.5709307300717366) q[10];
rz(-1.570797890857244) q[10];
ry(-2.6746281690154925) q[11];
rz(-1.5707976156496313) q[11];
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
ry(-3.141591675148834) q[0];
rz(0.13406079289988745) q[0];
ry(-9.323095628843703e-08) q[1];
rz(-1.8475508277965185) q[1];
ry(6.59154601063261e-07) q[2];
rz(-3.12985328697304) q[2];
ry(-6.816903255213447e-07) q[3];
rz(-0.35847988335566683) q[3];
ry(-1.9495719326398406) q[4];
rz(0.13235376800041193) q[4];
ry(0.4056190042153397) q[5];
rz(-2.1620594349260296) q[5];
ry(3.1415923994033355) q[6];
rz(0.4647186017474) q[6];
ry(-4.454197517489433e-07) q[7];
rz(0.9042645323825714) q[7];
ry(-3.1415851983382477) q[8];
rz(-1.3453885301295638) q[8];
ry(-1.489913314782805e-05) q[9];
rz(-2.1510458559230754) q[9];
ry(-1.5707993125569681) q[10];
rz(-0.8409790091214083) q[10];
ry(-1.570799797963261) q[11];
rz(-0.5914166420454019) q[11];