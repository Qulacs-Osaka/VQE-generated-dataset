OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.35073734334662304) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.27719358784030224) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.21626103784690265) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.5757285516333207) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.4919242849508533) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.123603907218155) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.16988228453612927) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.8966038553678525) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.20174957830602805) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.30994590435786956) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.0513096012657155) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(1.1938746545519812) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.05786109055798442) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.24900524203879382) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(1.568260933566611) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.04684264662943608) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.013545662925781765) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(-0.04158989692536143) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.0040431400870641) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.02173404592181258) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.26022255645107495) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.3450240092746938) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(-0.324635714422382) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-1.1179521643618269) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.25768851990855113) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(-1.1064603801814281) q[11];
cx q[10],q[11];
rx(-0.9116969429561917) q[0];
rz(0.0005849589383684091) q[0];
rx(-0.0022692084964160194) q[1];
rz(-0.8803917544691036) q[1];
rx(0.0003899320441774286) q[2];
rz(-0.018887205417241412) q[2];
rx(0.0004317490325184403) q[3];
rz(-0.2944541425930223) q[3];
rx(-0.00018893834455756595) q[4];
rz(-0.5162297710666153) q[4];
rx(-0.0005631926295431054) q[5];
rz(0.016814679882620996) q[5];
rx(-0.0001396810213009927) q[6];
rz(-0.25070099069678814) q[6];
rx(7.583018629356163e-05) q[7];
rz(0.5031589719158993) q[7];
rx(0.003841121291496433) q[8];
rz(0.04283959322624218) q[8];
rx(-0.021460924184601816) q[9];
rz(-0.5880519581857828) q[9];
rx(0.0004129150585616488) q[10];
rz(-0.807040117284252) q[10];
rx(0.013622757846607667) q[11];
rz(0.16718468258061334) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.9162785554141147) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.00016445719239236236) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.00020886695577083004) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.273161438555785) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.5107351405351984) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.0035438689401178862) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.02156479017220886) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.47834325624580526) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.5084721749547445) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.049089774329124924) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.05712492068635787) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.19892352392456158) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.037976472397033126) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.6991852107085436) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.019337568099708758) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(0.12019977746217633) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.012179584617922193) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.09300488772998317) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.021725047753943637) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.007623653257396485) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(0.610589669193731) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.056761206388904376) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(0.08229233596183891) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.9167367942874131) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(0.5620214850134919) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.14037083336048947) q[11];
cx q[10],q[11];
rx(-1.1981161343847613) q[0];
rz(-0.00023839030809002574) q[0];
rx(0.013793453719062015) q[1];
rz(-0.1729796350153123) q[1];
rx(-0.0023243969298309022) q[2];
rz(-0.6644754736028107) q[2];
rx(-1.7786232242006432) q[3];
rz(-0.00024579186151952887) q[3];
rx(0.00012538102276914094) q[4];
rz(-0.2412304600172829) q[4];
rx(-3.1411256729827093) q[5];
rz(-0.6112294682186169) q[5];
rx(0.0003723255008107779) q[6];
rz(-0.11592352124437562) q[6];
rx(-0.0006251710746604643) q[7];
rz(-1.9561608319888686) q[7];
rx(-0.003451788701871893) q[8];
rz(-0.588866875574575) q[8];
rx(0.00087875350874175) q[9];
rz(-0.2788016241979656) q[9];
rx(-0.0003502731455748063) q[10];
rz(-0.3737995872487978) q[10];
rx(-1.4151079074555135) q[11];
rz(-0.010987656110612016) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.02023755290577342) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0029198617569894296) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.0018262834129368713) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.6848990937478044) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0003565706409486724) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(3.040268519534988e-05) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(8.490209646484155e-05) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-4.856955119524651e-06) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.6874570486303797) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(5.275471925855532e-05) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.16536897569429967) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.31130390538423686) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.04821479258508887) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-1.2812213015148803) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.36220138848233246) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(0.07822600222138873) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(0.02481042320375774) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(-0.013897659957873553) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.03680732675436895) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.1833483958248084) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(1.1115971098819837) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.09759785080610933) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(0.04293118819167527) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(0.09148027433205165) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.005912503087564226) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.0008778046074586233) q[11];
cx q[10],q[11];
rx(-1.033187622844705) q[0];
rz(0.04970259712269051) q[0];
rx(-0.013781941613842968) q[1];
rz(-0.7537362346002512) q[1];
rx(0.0010045625619708551) q[2];
rz(-0.34020462085213393) q[2];
rx(-1.3626569751481854) q[3];
rz(-1.6003992944234195) q[3];
rx(0.0002835215148380882) q[4];
rz(-0.6203151163249481) q[4];
rx(7.966017872094572e-05) q[5];
rz(1.0526617709414914) q[5];
rx(-0.00015728839997758125) q[6];
rz(0.1483256721103731) q[6];
rx(-0.00016562164756535439) q[7];
rz(-0.016727917529832763) q[7];
rx(-0.0003812937892811518) q[8];
rz(0.14476540232499685) q[8];
rx(0.0004117707201829052) q[9];
rz(0.06810095173144626) q[9];
rx(0.0006072757843072928) q[10];
rz(-1.1543687482063576) q[10];
rx(-1.7379596579717649) q[11];
rz(0.3667435909451954) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.8872403458436501) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.6823685769909177) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.35328095111059493) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.3419334213700788) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.9176371990484982) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.5848254064560522) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.10836452453233848) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.14431265196497398) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.7585327291118327) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.5803941780352535) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.39075472514770876) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.27421428229503403) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.2131310201109411) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-1.4958245444207734) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-1.4260290222639624) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.2750288618554675) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.2251676065996347) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(-0.006261502419838147) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-1.188177502743347) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.8355186209507542) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.7652291890205033) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.44661313818452947) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(0.013254994178740107) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-1.311503542992744) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(0.18107318743574868) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.06913212134987803) q[11];
cx q[10],q[11];
rx(0.00648617573027409) q[0];
rz(0.023581605643207344) q[0];
rx(-0.0008296723652679628) q[1];
rz(-0.7439548181825423) q[1];
rx(0.000189065739717981) q[2];
rz(-0.013580805137518132) q[2];
rx(-0.00023726392101730884) q[3];
rz(-0.011138142766891004) q[3];
rx(5.835493128587468e-05) q[4];
rz(-0.019573598332788756) q[4];
rx(1.9459745331714957e-06) q[5];
rz(-0.00418315573197834) q[5];
rx(1.7687094061802142e-05) q[6];
rz(0.003834594304590378) q[6];
rx(0.00011640516243614828) q[7];
rz(-0.014551874580015202) q[7];
rx(7.354143533300149e-06) q[8];
rz(0.06941766832324106) q[8];
rx(5.272607184383979e-05) q[9];
rz(-0.00602379719671082) q[9];
rx(3.3756361893608587e-06) q[10];
rz(-0.09728741158298537) q[10];
rx(0.00979665660669526) q[11];
rz(-0.0018234629738169532) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.21473406314407106) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10499125335159787) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.8534238505573365) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2926196519755426) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.3080555029523917) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.022971778346122594) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.9691797785393858) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.003142222180759482) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.2537465154803165) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.06097423038810637) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.27219940258180897) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-1.4474234640513106) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.015215621332470064) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.2671837762314409) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.2351146359850979) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.8166638035899142) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-2.152920643954593) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.06509489399702914) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.515994348269241) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-1.124616741356379) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(0.9740810006138725) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-2.270497552199973) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(-0.03168533300444126) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(0.16246017660395018) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-2.3165951032591043) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(1.265672556452646) q[11];
cx q[10],q[11];
rx(-0.004928623391086088) q[0];
rz(0.014193886970506407) q[0];
rx(-0.0007200223840829527) q[1];
rz(-0.10876115226487494) q[1];
rx(-8.004480518657282e-05) q[2];
rz(-0.0056695217756372925) q[2];
rx(5.669024825443149e-05) q[3];
rz(0.07326855238878402) q[3];
rx(4.0114794723739e-06) q[4];
rz(0.0186226432718242) q[4];
rx(3.910028238834438e-05) q[5];
rz(-0.005573448161454334) q[5];
rx(-6.263780898640604e-06) q[6];
rz(-0.04064836968432852) q[6];
rx(-7.489847845568513e-06) q[7];
rz(-0.00010356810276929445) q[7];
rx(1.7408902439762654e-05) q[8];
rz(-0.03544581526939564) q[8];
rx(4.715990517143845e-06) q[9];
rz(0.0040616229708078785) q[9];
rx(-5.879250395297297e-05) q[10];
rz(-0.01340279057782339) q[10];
rx(0.00019426071265017187) q[11];
rz(-0.02684089129490604) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.2387963023834894) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.8903659552234534) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.19525464724461516) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.173832469421938) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.5793859862222441) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.2706142605648868) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.9386770255034994) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.12603095562160074) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.3692330744352837) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.6730956324028419) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.11449728065498373) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.008076778484822028) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.004157895721788114) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.16726117281741026) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.03054273210857375) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.6577388273166411) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.7076584796918343) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(-0.05060362936849965) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.4496852045353083) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(1.4029872815481543) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.08343859151841783) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.07179641349037436) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(0.030473598988047395) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.05331004145612545) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.06612720144777107) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(-1.2638017131353245) q[11];
cx q[10],q[11];
rx(0.0022501553715268134) q[0];
rz(-0.053734108541073126) q[0];
rx(4.9222277274190164e-05) q[1];
rz(0.025042809185304153) q[1];
rx(3.9643380293444386e-05) q[2];
rz(-0.05859915605740125) q[2];
rx(-0.00011843249961022398) q[3];
rz(-0.09505914944013337) q[3];
rx(5.8058312786579236e-05) q[4];
rz(-0.08570253183057677) q[4];
rx(-3.289667235532081e-05) q[5];
rz(-0.09553805738528864) q[5];
rx(1.553459360841763e-05) q[6];
rz(-0.0384365458196002) q[6];
rx(2.0794773855250836e-05) q[7];
rz(-0.06181842472363564) q[7];
rx(-3.372630609203793e-05) q[8];
rz(-0.0480954143214666) q[8];
rx(0.0001419754958385555) q[9];
rz(-0.052289471188532506) q[9];
rx(5.768192672069525e-05) q[10];
rz(-0.059716203867362684) q[10];
rx(0.0008034838058696991) q[11];
rz(-0.0457227258494411) q[11];