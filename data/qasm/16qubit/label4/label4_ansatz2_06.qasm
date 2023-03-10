OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(3.1415924902085446) q[0];
rz(1.4724463236795495) q[0];
ry(-1.5708045622608964) q[1];
rz(-1.0432123962089346) q[1];
ry(3.479340099089917e-08) q[2];
rz(-2.6549832198503265) q[2];
ry(1.5708276126224323) q[3];
rz(-3.1414462690941978) q[3];
ry(-4.6669133524706567e-07) q[4];
rz(2.5377510094087143) q[4];
ry(-1.5707843824475511) q[5];
rz(0.7646542730788289) q[5];
ry(1.570796210415529) q[6];
rz(-2.171903642949716e-06) q[6];
ry(1.5707954250564748) q[7];
rz(-1.1171137361279468) q[7];
ry(1.5707809316681978) q[8];
rz(-2.8053976104164087) q[8];
ry(-1.5707956342380236) q[9];
rz(4.827668852547015e-06) q[9];
ry(1.5707951145263004) q[10];
rz(1.8391571864309295) q[10];
ry(0.13596072442028806) q[11];
rz(3.1255340197167105) q[11];
ry(3.1415924683695495) q[12];
rz(-2.235130556612086) q[12];
ry(-3.1415919812478545) q[13];
rz(-0.13489256787778686) q[13];
ry(-3.1415925403071223) q[14];
rz(-2.76497560158682) q[14];
ry(-3.1415922114263455) q[15];
rz(-0.9168341549403863) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(3.1415914539499874) q[0];
rz(0.09442022233693681) q[0];
ry(-3.141592610305641) q[1];
rz(0.5275828344312601) q[1];
ry(6.31988354982127e-08) q[2];
rz(1.1333796716849935) q[2];
ry(-3.0372604711604856) q[3];
rz(0.00014697825875674894) q[3];
ry(-2.7464431351376106e-07) q[4];
rz(-1.4629933902223373) q[4];
ry(3.141591819605246) q[5];
rz(2.335452082077619) q[5];
ry(3.007516409778981) q[6];
rz(-0.6538916892753273) q[6];
ry(-3.1305296515721115) q[7];
rz(2.0245037458302986) q[7];
ry(3.1414793428317296) q[8];
rz(-0.5618369320693938) q[8];
ry(1.3373461877702493) q[9];
rz(-2.881867514262902) q[9];
ry(2.0677181655770482e-07) q[10];
rz(-0.1826565361066681) q[10];
ry(1.3923636359843877e-06) q[11];
rz(-1.655293906315797) q[11];
ry(1.5707964514022805) q[12];
rz(-1.3817676871153388) q[12];
ry(1.5707960679644684) q[13];
rz(-0.0929509498192278) q[13];
ry(1.570796000537146) q[14];
rz(3.1415924981554193) q[14];
ry(2.9222895365827557) q[15];
rz(-2.3752053874671555) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(1.5707960754545616) q[0];
rz(1.9601403014719425) q[0];
ry(2.379693378355882) q[1];
rz(3.141591759696533) q[1];
ry(-1.5707946441741942) q[2];
rz(-3.1415918851818336) q[2];
ry(1.5707880005415227) q[3];
rz(1.7849288809486008) q[3];
ry(2.1363500259853887e-07) q[4];
rz(-1.0400685766179585) q[4];
ry(-1.4725317071602033) q[5];
rz(-0.4509338628736757) q[5];
ry(-3.1415925663838506) q[6];
rz(0.9169069259129629) q[6];
ry(1.5707963365128563) q[7];
rz(-1.663132410944522) q[7];
ry(3.1415690802392926) q[8];
rz(1.015952028988986) q[8];
ry(3.1415897543620397) q[9];
rz(-0.8044759918333417) q[9];
ry(-1.404170661167825e-05) q[10];
rz(-2.683979696952646) q[10];
ry(-9.401486122939673e-07) q[11];
rz(-2.6127756895571945) q[11];
ry(-3.1415923637474132) q[12];
rz(-0.47593362369584524) q[12];
ry(3.141458970867996) q[13];
rz(-2.4777660307176834) q[13];
ry(-1.57092935317467) q[14];
rz(1.5707962915243512) q[14];
ry(-3.1415921879967237) q[15];
rz(2.337185710281472) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-3.1415926002040346) q[0];
rz(1.0641894422117413) q[0];
ry(1.5707963250259716) q[1];
rz(-0.7206503010833575) q[1];
ry(0.7853862989709324) q[2];
rz(-1.570794505347867) q[2];
ry(1.5707969313529733) q[3];
rz(-7.457542444022636e-06) q[3];
ry(1.570796772470542) q[4];
rz(-1.6130217659338084e-06) q[4];
ry(3.1415895755559244) q[5];
rz(2.6909150697678808) q[5];
ry(1.5707965391062686) q[6];
rz(-1.020699980039601) q[6];
ry(1.5707963608657272) q[7];
rz(3.141591836815718) q[7];
ry(3.2739328628780073e-06) q[8];
rz(2.7984038621817855) q[8];
ry(3.1415889533897627) q[9];
rz(0.5065845568513936) q[9];
ry(3.141590926273906) q[10];
rz(0.5433068611650579) q[10];
ry(-6.457527081948911e-07) q[11];
rz(-1.6790736545335694) q[11];
ry(3.141590424833329) q[12];
rz(0.9058341193673434) q[12];
ry(-1.5285678388164499) q[13];
rz(-1.21451606906568) q[13];
ry(1.5093300432629952) q[14];
rz(-1.5707962100746151) q[14];
ry(-1.570796187520954) q[15];
rz(0.060076007200842205) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(3.1415916000235242) q[0];
rz(-0.7235953109819269) q[0];
ry(3.1415897340973675) q[1];
rz(-1.6942252519603624) q[1];
ry(-2.356200639166787) q[2];
rz(1.5708021139444366) q[2];
ry(0.8183654106620676) q[3];
rz(-3.1415877567276533) q[3];
ry(2.3562667058277342) q[4];
rz(1.570794103512113) q[4];
ry(2.9616674049146536) q[5];
rz(-1.5999134147933778) q[5];
ry(2.23903802165637e-08) q[6];
rz(-0.41005369062309827) q[6];
ry(-1.5378235736345847) q[7];
rz(-1.3832863717466576e-06) q[7];
ry(0.7238697695236018) q[8];
rz(1.5707848266865518) q[8];
ry(-1.0094223671074003) q[9];
rz(-1.7807003561636536) q[9];
ry(-2.8216652756672866) q[10];
rz(0.6125904232483158) q[10];
ry(-3.141592126056525) q[11];
rz(0.1634354264973847) q[11];
ry(1.44206308129799) q[12];
rz(-1.8022476014320823) q[12];
ry(3.1415925319285476) q[13];
rz(1.0374049861715928) q[13];
ry(-1.4420635049182087) q[14];
rz(-1.5137811902965168) q[14];
ry(3.141524621838228) q[15];
rz(-0.5724952376286727) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(2.1754206507296203e-06) q[0];
rz(-1.6197693006328115) q[0];
ry(-1.5707961029589261) q[1];
rz(1.5707932806496139) q[1];
ry(0.7854007364872072) q[2];
rz(1.5475609417916216) q[2];
ry(1.570781389773708) q[3];
rz(-2.3570455255417784) q[3];
ry(-0.7853982453438602) q[4];
rz(1.528314016215539) q[4];
ry(-0.0003945478911528965) q[5];
rz(-1.5414274683059577) q[5];
ry(-1.5225899041755506e-08) q[6];
rz(1.867349864555309) q[6];
ry(-1.5707964784180113) q[7];
rz(-1.571647422067871) q[7];
ry(2.355784809568761) q[8];
rz(1.9943164208729334) q[8];
ry(3.141578497086139) q[9];
rz(-1.7807055886210037) q[9];
ry(3.1415899383891857) q[10];
rz(-2.5290135245321888) q[10];
ry(-3.1415917878024553) q[11];
rz(-2.005793613911248) q[11];
ry(-3.1415913120027543) q[12];
rz(1.3393451589456882) q[12];
ry(3.1415922822654396) q[13];
rz(0.7257991272194033) q[13];
ry(-1.7724901353371302e-06) q[14];
rz(1.5137806182133413) q[14];
ry(3.141592375105596) q[15];
rz(-2.203367403310586) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-1.111232002736829e-05) q[0];
rz(-1.6325229436996942) q[0];
ry(1.5707954263222537) q[1];
rz(-1.8655485373202607) q[1];
ry(1.1360033575974852e-05) q[2];
rz(1.6517414233622538) q[2];
ry(-1.570795242515504) q[3];
rz(-5.670315319505903e-07) q[3];
ry(3.141588325481518) q[4];
rz(-0.042485325712823256) q[4];
ry(1.7976627632924886) q[5];
rz(-2.3673428008585597e-07) q[5];
ry(6.588983269619715e-08) q[6];
rz(2.7214405571822766) q[6];
ry(-1.5707967893706607) q[7];
rz(-2.772828112839469) q[7];
ry(3.1415923267045622) q[8];
rz(1.9943328123598005) q[8];
ry(-1.7011851266246394) q[9];
rz(1.5707982775527833) q[9];
ry(-0.09647694991726216) q[10];
rz(-3.1010492349070655) q[10];
ry(-3.1415926269155556) q[11];
rz(1.22551342113615) q[11];
ry(0.656059231971251) q[12];
rz(-1.5707961302907996) q[12];
ry(-1.5707959885763523) q[13];
rz(1.5707963611291162) q[13];
ry(-2.2268550115598345) q[14];
rz(1.4802404248643528) q[14];
ry(-1.57079605149538) q[15];
rz(1.5708004998848006) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(5.730958087891234e-05) q[0];
rz(-1.7132781468908176) q[0];
ry(1.8139181341814804e-06) q[1];
rz(0.0017403151748514034) q[1];
ry(5.6726401584050734e-05) q[2];
rz(-2.6221131186203714) q[2];
ry(1.5707964439271225) q[3];
rz(-1.5707972153818162) q[3];
ry(-1.5707966082499103) q[4];
rz(1.57079566499951) q[4];
ry(-1.5707965616777697) q[5];
rz(-0.3279961838283337) q[5];
ry(-3.141589813219102) q[6];
rz(-0.17175523331075307) q[6];
ry(-3.1415914207718196) q[7];
rz(1.0048797851153752) q[7];
ry(-2.3828793924894884) q[8];
rz(1.5707953886306054) q[8];
ry(-2.6432894786786423) q[9];
rz(1.5707954911309443) q[9];
ry(4.2257317645066905e-08) q[10];
rz(-0.040520786018234674) q[10];
ry(-1.8441324952532137e-07) q[11];
rz(1.756559607295869) q[11];
ry(1.5707957897715392) q[12];
rz(-0.7259086698625581) q[12];
ry(1.5707965796822476) q[13];
rz(2.698059905465228) q[13];
ry(3.1415917172726746) q[14];
rz(0.3182104659311165) q[14];
ry(-1.5707964332627078) q[15];
rz(2.9029749284518926) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-3.141592574814565) q[0];
rz(3.060767369500407) q[0];
ry(-3.1415921337704833) q[1];
rz(-0.3441126188258591) q[1];
ry(-5.353821374048967e-08) q[2];
rz(0.9936041805645545) q[2];
ry(-1.5707717739129592) q[3];
rz(1.5707962222280125) q[3];
ry(-1.5706558455991895) q[4];
rz(0.15623938694770262) q[4];
ry(-3.141580695381715) q[5];
rz(2.6120762936708264) q[5];
ry(-1.5707965227443506) q[6];
rz(-0.2531119897658405) q[6];
ry(-3.1414543895476688) q[7];
rz(0.6361157023255002) q[7];
ry(1.7261495141868117) q[8];
rz(-3.1415925034710663) q[8];
ry(1.5900081316102423) q[9];
rz(1.4047188228257097) q[9];
ry(-1.745362283114497) q[10];
rz(3.1415925909455082) q[10];
ry(-1.2488891707462813e-06) q[11];
rz(3.022662975049664) q[11];
ry(-3.1415924247957823) q[12];
rz(-2.2967040305400217) q[12];
ry(-3.1415920685896923) q[13];
rz(2.7487416870611976) q[13];
ry(4.100424877787915e-07) q[14];
rz(-1.979562462892953) q[14];
ry(3.141591982530766) q[15];
rz(1.332178051665436) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(1.5707967219207297) q[0];
rz(1.0199025089093448) q[0];
ry(-1.7359433487391385) q[1];
rz(-0.1869315159001274) q[1];
ry(-1.57079628530679) q[2];
rz(-0.601993770800398) q[2];
ry(1.8589323858525049) q[3];
rz(1.3838677750283654) q[3];
ry(-3.1415923582522582) q[4];
rz(-2.1021837927251763) q[4];
ry(-3.0164638431616866) q[5];
rz(-0.07379224571450571) q[5];
ry(3.141592403517577) q[6];
rz(-2.5256334988279923) q[6];
ry(1.4784231760340507) q[7];
rz(1.3838663053377118) q[7];
ry(1.5707971780461119) q[8];
rz(2.60824701603524) q[8];
ry(2.285245911634712) q[9];
rz(1.2744774924448778) q[9];
ry(-1.5707959063149621) q[10];
rz(-1.3828276957744172) q[10];
ry(-1.5707967156744063) q[11];
rz(-0.18692985076477017) q[11];
ry(1.5707962616462625) q[12];
rz(-2.7583166092271263) q[12];
ry(1.5707966478417417) q[13];
rz(1.3838702160151906) q[13];
ry(-1.5707961383960252) q[14];
rz(1.95407244053661) q[14];
ry(1.5707947937695066) q[15];
rz(-0.18692447449739813) q[15];