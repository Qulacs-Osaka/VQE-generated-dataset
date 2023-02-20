OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.5707831478874779) q[0];
rz(0.007821278798307739) q[0];
ry(-3.1400210413807015) q[1];
rz(0.3438803070718687) q[1];
ry(-1.5704968239799673) q[2];
rz(1.3670674201312415e-05) q[2];
ry(-3.1415324624977607) q[3];
rz(1.710189620132076) q[3];
ry(1.570788396575864) q[4];
rz(-0.0002431018067581522) q[4];
ry(1.569595563746863) q[5];
rz(-1.675488015312379) q[5];
ry(1.3786864729219994) q[6];
rz(-1.8327387764759147) q[6];
ry(3.1255149489758702) q[7];
rz(-1.5657300442944984) q[7];
ry(0.004192767085361571) q[8];
rz(2.9856717264832526) q[8];
ry(-3.111987533903302) q[9];
rz(-1.5674432934820055) q[9];
ry(-3.0601031362209596e-06) q[10];
rz(-2.6527795867305715) q[10];
ry(0.00926529115479752) q[11];
rz(0.750308113426307) q[11];
ry(3.0750210057206573) q[12];
rz(-1.9478704985194497) q[12];
ry(0.0007775183045415368) q[13];
rz(-1.7010558188104705) q[13];
ry(-1.570537812012884) q[14];
rz(-3.0965647783759453) q[14];
ry(3.128484735046389) q[15];
rz(-3.1358948175453625) q[15];
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
ry(-0.01589686357090255) q[0];
rz(1.562943911316998) q[0];
ry(1.570741019180816) q[1];
rz(-0.0001468565493863978) q[1];
ry(1.7537345511081959) q[2];
rz(1.109917001036787) q[2];
ry(1.5710627047260586) q[3];
rz(-3.136833490889237) q[3];
ry(-0.052503336040619075) q[4];
rz(1.5711735677293825) q[4];
ry(1.620160536468974) q[5];
rz(-1.1220247225035742) q[5];
ry(0.0003674265496440343) q[6];
rz(-3.0414218633312027) q[6];
ry(3.033285249987865) q[7];
rz(0.0038314030448765024) q[7];
ry(-0.0032889735342109816) q[8];
rz(3.0875488023814266) q[8];
ry(0.020874858935641782) q[9];
rz(-0.0021384313142279567) q[9];
ry(3.141585422006947) q[10];
rz(-2.907472664965735) q[10];
ry(-0.457576882433047) q[11];
rz(3.1408260467922466) q[11];
ry(0.03465591361103648) q[12];
rz(-0.18031570673347083) q[12];
ry(-1.5711147229886868) q[13];
rz(2.5385865810747403) q[13];
ry(0.7096438657592232) q[14];
rz(2.328956896673127) q[14];
ry(-1.013972344280852) q[15];
rz(4.9475015795330535e-05) q[15];
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
ry(2.279159669692472) q[0];
rz(1.5707719971667098) q[0];
ry(2.0852006381387374) q[1];
rz(-1.5632820418340305) q[1];
ry(3.141583401005) q[2];
rz(1.109809870768622) q[2];
ry(0.09153601770266606) q[3];
rz(1.1077848445253613) q[3];
ry(0.8464478114496851) q[4];
rz(-2.755375365255489) q[4];
ry(-0.031741791977416604) q[5];
rz(1.3861063505285687) q[5];
ry(3.1405401823439782) q[6];
rz(1.409429438281523) q[6];
ry(-0.5975393362565605) q[7];
rz(0.01663598628165144) q[7];
ry(-3.131167337379079) q[8];
rz(-3.140984150021449) q[8];
ry(1.5870616905920247) q[9];
rz(-6.406555409732562e-06) q[9];
ry(-3.141587995324223) q[10];
rz(0.6893856181661606) q[10];
ry(-2.343081108430742) q[11];
rz(-0.03281412560934757) q[11];
ry(-0.005769420405569825) q[12];
rz(-2.9632896442936687) q[12];
ry(3.1002727308331766) q[13];
rz(-2.2897070911583106) q[13];
ry(-3.194141472245191e-05) q[14];
rz(0.7783668196458261) q[14];
ry(0.3630542932590447) q[15];
rz(-0.12950572383632153) q[15];
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
ry(-0.9076315588992889) q[0];
rz(1.5706239956938157) q[0];
ry(3.140776842720829) q[1];
rz(-1.7778541556303233) q[1];
ry(0.6880941000367495) q[2];
rz(1.570888458429998) q[2];
ry(-1.2494835774262006e-05) q[3];
rz(0.7495217755612538) q[3];
ry(-3.141585010466195) q[4];
rz(-2.0150698813662213) q[4];
ry(0.0007617660679205635) q[5];
rz(-2.957429144307824) q[5];
ry(-1.5404609195911476) q[6];
rz(2.91421798373395) q[6];
ry(-3.077694644446625) q[7];
rz(2.3122255568925723) q[7];
ry(-1.8946365441400719) q[8];
rz(0.9160559621897848) q[8];
ry(1.5556665356305688) q[9];
rz(1.7884388341491546) q[9];
ry(1.570790127641227) q[10];
rz(1.005268973066423e-05) q[10];
ry(-0.034264307751071) q[11];
rz(-2.83925018771985) q[11];
ry(-0.32213456587472766) q[12];
rz(1.4805808535222869) q[12];
ry(-3.14073390469111) q[13];
rz(-1.6900643902588444) q[13];
ry(0.8064759705495579) q[14];
rz(1.5754080653880278) q[14];
ry(3.1280466499087587) q[15];
rz(-2.989383407068964) q[15];
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
ry(-0.2877503872241774) q[0];
rz(-1.569708415902463) q[0];
ry(1.5661500033412192) q[1];
rz(1.809755841293012) q[1];
ry(1.2867510823652306) q[2];
rz(1.900816219758318) q[2];
ry(-3.140170985409523) q[3];
rz(-0.28106493943546607) q[3];
ry(-2.899012894615715e-05) q[4];
rz(-2.312243863220089) q[4];
ry(1.5706211195652564) q[5];
rz(1.5706725873020504) q[5];
ry(-1.3178396425716917e-06) q[6];
rz(2.8018077441256297) q[6];
ry(1.2999065276630972e-05) q[7];
rz(-0.7274998941640552) q[7];
ry(-1.4404098003240051e-05) q[8];
rz(1.5433070308882977) q[8];
ry(-3.141583948867466) q[9];
rz(0.2202607051218891) q[9];
ry(-1.5707711048873354) q[10];
rz(2.190066827633097) q[10];
ry(-3.141569454329403) q[11];
rz(-1.2996107435486879) q[11];
ry(-4.496929380492309e-06) q[12];
rz(2.6554466555366916) q[12];
ry(3.1364181840028267) q[13];
rz(1.5362527754471154) q[13];
ry(-3.137906110787576) q[14];
rz(1.5755043863504639) q[14];
ry(3.1415569328559) q[15];
rz(1.8532809322884944) q[15];
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
ry(-0.0013783678372885078) q[0];
rz(-1.5717888975178211) q[0];
ry(3.140492121463171) q[1];
rz(0.23789049365752732) q[1];
ry(3.1415237988383735) q[2];
rz(-1.6852008912456524) q[2];
ry(-1.5196999038138745) q[3];
rz(-3.1262025545301366) q[3];
ry(0.13750394442576755) q[4];
rz(2.0596484249194056) q[4];
ry(-1.5720341890284288) q[5];
rz(0.00015295489981756837) q[5];
ry(3.136768406573908) q[6];
rz(1.4809670181907026) q[6];
ry(2.7431356608292927) q[7];
rz(1.8804457740825713) q[7];
ry(-0.00027272172878030923) q[8];
rz(1.31450989822228) q[8];
ry(2.966484462799219) q[9];
rz(1.3111439456355636) q[9];
ry(3.140084773151985) q[10];
rz(-1.0093984207224365) q[10];
ry(-0.12961193110072336) q[11];
rz(-0.7994521042013147) q[11];
ry(0.0009950404685676872) q[12];
rz(-0.5092848974568414) q[12];
ry(-1.7191154237756558) q[13];
rz(-1.5735830405775184) q[13];
ry(-1.5793762886954668) q[14];
rz(-0.4185119944010366) q[14];
ry(3.0645794012296523) q[15];
rz(-1.6011390625249964) q[15];
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
ry(-0.1141285070031941) q[0];
rz(-0.9006212338310808) q[0];
ry(-1.570940857832706) q[1];
rz(3.1415112504201823) q[1];
ry(0.00013602091364095514) q[2];
rz(0.43646988665429726) q[2];
ry(-1.5707058651459072) q[3];
rz(1.5708735045848679) q[3];
ry(-0.1289425200744737) q[4];
rz(-0.4841854623013641) q[4];
ry(1.571879927342489) q[5];
rz(3.1415658162447966) q[5];
ry(-3.1415516534931958) q[6];
rz(0.8056241227400962) q[6];
ry(-3.1414969962248334) q[7];
rz(-1.3258271136787398) q[7];
ry(-3.141444786708801) q[8];
rz(-1.0200622107125465) q[8];
ry(7.814686612306332e-05) q[9];
rz(-1.353848885782039) q[9];
ry(-3.141468245748451) q[10];
rz(0.18159058125393118) q[10];
ry(3.1415177831221666) q[11];
rz(-0.8156074877378925) q[11];
ry(-1.9669217286022445e-05) q[12];
rz(2.190776059997985) q[12];
ry(1.5707017375725325) q[13];
rz(-1.5708494035709097) q[13];
ry(8.505921168122654e-05) q[14];
rz(-2.686661943088303) q[14];
ry(3.1414665984207577) q[15];
rz(0.425714251257594) q[15];
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
ry(-3.1382386240489852) q[0];
rz(2.240829651500358) q[0];
ry(1.5701330153840742) q[1];
rz(1.5730856481375814) q[1];
ry(-0.3309740342544777) q[2];
rz(-3.1332419111698537) q[2];
ry(-1.583237288669207) q[3];
rz(0.3455113672462045) q[3];
ry(1.5707993947327517) q[4];
rz(1.5609196899444546) q[4];
ry(1.5700259103805574) q[5];
rz(-1.433086584251292) q[5];
ry(1.570592894358728) q[6];
rz(-1.5703350561490006) q[6];
ry(-3.1326160888978936) q[7];
rz(0.7608422514373162) q[7];
ry(3.1412178812836675) q[8];
rz(2.737128154539717) q[8];
ry(0.010100980166819675) q[9];
rz(-1.4812965859895089) q[9];
ry(-6.801100351509931e-06) q[10];
rz(-0.0558511984488201) q[10];
ry(3.1346493308518886) q[11];
rz(-0.3880323479078005) q[11];
ry(-3.1415627871145264) q[12];
rz(2.518606719461251) q[12];
ry(0.7118067867697387) q[13];
rz(1.3714149669763513) q[13];
ry(0.010216182380107242) q[14];
rz(-1.6071029639176926) q[14];
ry(-0.00023080143958183593) q[15];
rz(1.5713690450855315) q[15];
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
ry(-1.5630799277358995) q[0];
rz(-1.913382314180907) q[0];
ry(3.141392106259154) q[1];
rz(1.571769432730647) q[1];
ry(-1.5707116949019442) q[2];
rz(0.6381827969884569) q[2];
ry(3.141575728439248) q[3];
rz(0.3454041288018416) q[3];
ry(-1.570791310349441) q[4];
rz(-3.141514549705003) q[4];
ry(2.8421820121906074e-05) q[5];
rz(-1.7088480941451143) q[5];
ry(1.570794966512389) q[6];
rz(-1.5054576470391048) q[6];
ry(-3.1415005475096343) q[7];
rz(2.4112393518884847) q[7];
ry(8.829957566902635e-06) q[8];
rz(2.210510841479043) q[8];
ry(3.141564424606274) q[9];
rz(-1.1769880245237465) q[9];
ry(-3.1415876405624044) q[10];
rz(2.116360409926424) q[10];
ry(3.141549766411593) q[11];
rz(2.845165878401378) q[11];
ry(3.1415683602182325) q[12];
rz(-2.5268251002954925) q[12];
ry(0.00021513105475445826) q[13];
rz(-1.3714957001256782) q[13];
ry(1.563165120227585) q[14];
rz(1.5711297062231118) q[14];
ry(4.8850383613796566e-05) q[15];
rz(-2.8058209878125706) q[15];
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
ry(-6.830917603305013e-07) q[0];
rz(-2.969486779539302) q[0];
ry(-1.570666088807875) q[1];
rz(-1.6005013116780056) q[1];
ry(-4.5063058412655225e-05) q[2];
rz(-2.268796904912702) q[2];
ry(-1.5706799203820827) q[3];
rz(-0.019371031693812494) q[3];
ry(1.5707712630513493) q[4];
rz(-1.522538890732946) q[4];
ry(-1.5708438303187597) q[5];
rz(-0.05189876387830683) q[5];
ry(-3.141523778064374) q[6];
rz(3.1196742202493164) q[6];
ry(3.141515126208651) q[7];
rz(-1.4842970697985516) q[7];
ry(-3.1415477846109896) q[8];
rz(1.7772741298918309) q[8];
ry(3.1415744242663037) q[9];
rz(2.670629673144856) q[9];
ry(-1.868950770645005e-05) q[10];
rz(-2.076893517846736) q[10];
ry(-0.00010069224323033387) q[11];
rz(-2.92010868305018) q[11];
ry(-4.6925863495417275e-05) q[12];
rz(1.1109106131017006) q[12];
ry(1.5703703208253588) q[13];
rz(-0.8394898069539423) q[13];
ry(1.570646576198902) q[14];
rz(2.3209957047343106e-06) q[14];
ry(3.1413930259685245) q[15];
rz(1.6982705831312144) q[15];
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
ry(3.1163115180918437) q[0];
rz(-0.17142478935325678) q[0];
ry(-0.0790117724101785) q[1];
rz(-1.1380207732828236) q[1];
ry(-3.0281385566199206) q[2];
rz(-3.056800848546238) q[2];
ry(-0.11954411584987784) q[3];
rz(-0.07157516402526415) q[3];
ry(-3.0243515245675767) q[4];
rz(0.068104254422351) q[4];
ry(-1.5717644775629722) q[5];
rz(1.605684520759085) q[5];
ry(3.005411415428106) q[6];
rz(3.0403191855032228) q[6];
ry(3.1106798783280962) q[7];
rz(0.0455463860527253) q[7];
ry(3.0241753939636715) q[8];
rz(-0.11106504356993864) q[8];
ry(0.004459908316576389) q[9];
rz(0.7828994057531515) q[9];
ry(-2.3279752308984) q[10];
rz(-1.670162049449643) q[10];
ry(-3.1234623758238667) q[11];
rz(0.3237340290215375) q[11];
ry(-0.08385025716953246) q[12];
rz(-0.31172603700307555) q[12];
ry(3.1085887377054156) q[13];
rz(2.314029555914908) q[13];
ry(-1.5708009860288858) q[14];
rz(-1.2651214694690194) q[14];
ry(0.033632982227696395) q[15];
rz(-0.8901664136071554) q[15];
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
ry(-1.5616436387716328) q[0];
rz(-1.4091976719027963) q[0];
ry(-0.00120485844939048) q[1];
rz(1.8552953146675384) q[1];
ry(0.006516948943299773) q[2];
rz(-2.338611799511815) q[2];
ry(3.1343183852838) q[3];
rz(-1.290330912142105) q[3];
ry(3.1285176134360597) q[4];
rz(-2.7524598482580456) q[4];
ry(3.125089190760867) q[5];
rz(1.4888287106375913) q[5];
ry(0.017629881384911856) q[6];
rz(-2.441955656760251) q[6];
ry(3.0998433219835313) q[7];
rz(-3.1021600263176428) q[7];
ry(1.62515701534035) q[8];
rz(-2.8823386614122364) q[8];
ry(0.03616505582119235) q[9];
rz(-0.34442868053108455) q[9];
ry(1.6191712445350843) q[10];
rz(-1.429450917513293) q[10];
ry(3.0971562422485777) q[11];
rz(3.140848444626173) q[11];
ry(1.5731771108959465) q[12];
rz(2.445062696233163) q[12];
ry(0.09036460782084532) q[13];
rz(3.0620430289778304) q[13];
ry(-1.5708584834276733) q[14];
rz(0.00011544284266662431) q[14];
ry(0.13628931202293332) q[15];
rz(3.104837453218436) q[15];
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
ry(3.141502629842437) q[0];
rz(0.6633881122301581) q[0];
ry(-3.141505566130292) q[1];
rz(-0.3574949640752949) q[1];
ry(3.141490933372382) q[2];
rz(-0.12108997562704717) q[2];
ry(-3.101845499544055e-05) q[3];
rz(-2.9873094681021652) q[3];
ry(3.14133974282846) q[4];
rz(-0.6990223044662238) q[4];
ry(-3.1415904688292886) q[5];
rz(0.40926701581325187) q[5];
ry(-3.141356336629741) q[6];
rz(2.759374276345947) q[6];
ry(2.3926637395737405e-05) q[7];
rz(-1.110297778546306) q[7];
ry(-3.141584509215215) q[8];
rz(-0.8085472530921822) q[8];
ry(3.1415847705084343) q[9];
rz(1.7172084803311822) q[9];
ry(3.141509365462339) q[10];
rz(0.6443722068939958) q[10];
ry(3.1415271513114176) q[11];
rz(2.0700933941889113) q[11];
ry(3.1413928327510487) q[12];
rz(1.3774155995295239) q[12];
ry(0.00014274214579204669) q[13];
rz(-0.9761125866632948) q[13];
ry(-1.5709532780132014) q[14];
rz(2.073930376658282) q[14];
ry(0.00015370920548597692) q[15];
rz(-1.0220532385186212) q[15];