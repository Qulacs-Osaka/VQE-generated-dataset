OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-1.5707649258401883) q[0];
rz(0.7896858190946687) q[0];
ry(-1.5707881495818425) q[1];
rz(-6.12342841311085e-05) q[1];
ry(-1.570796033277027) q[2];
rz(1.5707952152140012) q[2];
ry(1.5683811806973997) q[3];
rz(0.2032474107706461) q[3];
ry(1.5707987808333894) q[4];
rz(0.03694570814748221) q[4];
ry(2.2958354188418184e-05) q[5];
rz(-2.948781102523079) q[5];
ry(-0.014084943894529188) q[6];
rz(-1.5590302180851299) q[6];
ry(-5.674384797137401e-06) q[7];
rz(0.3049193278744326) q[7];
ry(6.0794724542567735e-05) q[8];
rz(-0.44362598333870995) q[8];
ry(-0.004388687871307031) q[9];
rz(-2.923458049791332) q[9];
ry(-1.5739843674653287) q[10];
rz(-1.570785848853964) q[10];
ry(-3.1396413310105147) q[11];
rz(1.7216455007015583) q[11];
ry(0.0021319104228932017) q[12];
rz(2.90950720325105) q[12];
ry(-0.00017719846516861537) q[13];
rz(1.6749201519077381) q[13];
ry(1.5727236956931572) q[14];
rz(-1.1306612686986908) q[14];
ry(-1.569698879680091) q[15];
rz(-1.5708825303976557) q[15];
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
ry(0.0005934526215112257) q[0];
rz(0.7771115704405984) q[0];
ry(0.531116917564959) q[1];
rz(1.6045897706639947) q[1];
ry(-1.5707984502388053) q[2];
rz(-3.0601990316571235) q[2];
ry(-1.4742564152763804) q[3];
rz(-0.8592838524667288) q[3];
ry(-0.0009309874103416849) q[4];
rz(1.533851905724939) q[4];
ry(1.5708151681603346) q[5];
rz(1.9517206383818122) q[5];
ry(-1.4942700169810899) q[6];
rz(-0.6099035039840617) q[6];
ry(-1.5708209998865734) q[7];
rz(7.352704746687921e-05) q[7];
ry(-2.442132958374758) q[8];
rz(-2.1946086939679246) q[8];
ry(0.01291741708165528) q[9];
rz(-1.7435985285946611) q[9];
ry(-1.5707960679640978) q[10];
rz(-0.7910379597209788) q[10];
ry(-1.060256851936505) q[11];
rz(1.279378376588966) q[11];
ry(1.0779332290528814) q[12];
rz(-2.9814492430828943) q[12];
ry(1.570810740256122) q[13];
rz(1.5443605639894153) q[13];
ry(-0.00021217775958692897) q[14];
rz(0.03820366216543559) q[14];
ry(1.5707735905829816) q[15];
rz(-1.632416517944474) q[15];
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
ry(0.011271102305270292) q[0];
rz(-1.566795142376161) q[0];
ry(0.0029490108275866864) q[1];
rz(1.5371014821367803) q[1];
ry(-3.1415924537426005) q[2];
rz(0.08973218093461624) q[2];
ry(-3.141583338723969) q[3];
rz(-2.495078403212795) q[3];
ry(2.6361344737505035) q[4];
rz(1.570795962842173) q[4];
ry(-3.141591668070765) q[5];
rz(0.38092913700546394) q[5];
ry(3.140830046312299) q[6];
rz(-2.276075859807192) q[6];
ry(-0.5557151086114018) q[7];
rz(1.570714317481634) q[7];
ry(-3.1415875217203824) q[8];
rz(2.1138404396227712) q[8];
ry(-3.141445325462958) q[9];
rz(-1.7853879767486387) q[9];
ry(-3.1296217086866926) q[10];
rz(-0.7011235434911409) q[10];
ry(0.0029147934402349333) q[11];
rz(2.1048918174809277) q[11];
ry(-0.0016562694403704514) q[12];
rz(1.4149895791767149) q[12];
ry(1.545577248596131) q[13];
rz(-0.5089975036960392) q[13];
ry(-0.571821562598843) q[14];
rz(-1.9818180141515331) q[14];
ry(3.1415906836277006) q[15];
rz(-0.32375800283450096) q[15];
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
ry(2.514400876703203) q[0];
rz(-0.04887555930657684) q[0];
ry(2.6462191233438817) q[1];
rz(1.5709484099249647) q[1];
ry(-7.434403244943941e-07) q[2];
rz(1.697759194480481) q[2];
ry(-3.141591078057513) q[3];
rz(2.91709023791858) q[3];
ry(1.7070068053039957) q[4];
rz(-1.57079278296393) q[4];
ry(-1.6144471592757421) q[5];
rz(1.5706852852683064) q[5];
ry(-0.015856108116195446) q[6];
rz(1.8783039952865899) q[6];
ry(1.1074222021565918) q[7];
rz(-1.5532437255927523) q[7];
ry(-8.098202458839411e-06) q[8];
rz(3.050513030536941) q[8];
ry(3.1415479170932477) q[9];
rz(1.5956320786265725) q[9];
ry(-1.5392987386952935e-05) q[10];
rz(2.231248711576221) q[10];
ry(-3.14086729485696) q[11];
rz(0.8297519134327462) q[11];
ry(0.5450460983477452) q[12];
rz(-3.1325222813401328) q[12];
ry(3.1368231790005643) q[13];
rz(1.6196384749370116) q[13];
ry(-1.7588720710792014) q[14];
rz(1.5708030385516738) q[14];
ry(-3.000859330222634e-05) q[15];
rz(2.2929122750467696) q[15];
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
ry(-3.1415415827592024) q[0];
rz(3.0232300033180644) q[0];
ry(2.4469899963722876) q[1];
rz(-1.6958943325971987) q[1];
ry(3.1415916469111043) q[2];
rz(-1.6503888302922016) q[2];
ry(3.1415704542515255) q[3];
rz(1.7307702724122649) q[3];
ry(-2.4345941481081574) q[4];
rz(2.222023211633086) q[4];
ry(0.17851621281750152) q[5];
rz(-1.5705668293553805) q[5];
ry(-0.03298911368661181) q[6];
rz(-1.7923871245537413) q[6];
ry(-2.4589793308023618) q[7];
rz(0.0874590364317749) q[7];
ry(3.1415863143987264) q[8];
rz(1.8785000513762498) q[8];
ry(2.900903175462704) q[9];
rz(0.8561377899683595) q[9];
ry(1.6649002624300873) q[10];
rz(0.28332757971959577) q[10];
ry(-2.7499667139053776) q[11];
rz(-0.7863124838567593) q[11];
ry(-0.06284268980415496) q[12];
rz(-2.91521685778608) q[12];
ry(-1.5605855067545296) q[13];
rz(3.1023222639635444) q[13];
ry(1.5707259491093533) q[14];
rz(-2.5976205799410574) q[14];
ry(3.1415919531874446) q[15];
rz(2.6865850486062746) q[15];
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
ry(3.1415311301226687) q[0];
rz(-2.0941482314042754) q[0];
ry(1.0180648157387395e-05) q[1];
rz(1.6960807163818814) q[1];
ry(-3.1415910349865315) q[2];
rz(3.036871103623276) q[2];
ry(-3.1415855276617384) q[3];
rz(-1.4166394301930172) q[3];
ry(3.868183124916148e-06) q[4];
rz(-0.6229552913597852) q[4];
ry(-1.570915146440636) q[5];
rz(1.5712360707915662) q[5];
ry(-3.1408764306224692) q[6];
rz(-1.5946077440228497) q[6];
ry(3.1415672640778243) q[7];
rz(-3.078610732059309) q[7];
ry(-3.1415912288022914) q[8];
rz(-2.1220512950517247) q[8];
ry(-3.141329470255756) q[9];
rz(3.0675382232625394) q[9];
ry(-1.5751588046782012) q[10];
rz(3.1265559840627732) q[10];
ry(-6.202684448227657e-05) q[11];
rz(-2.4282851668595056) q[11];
ry(3.1412304775567423) q[12];
rz(0.25633646413246947) q[12];
ry(3.1402708349665196) q[13];
rz(-0.057673863420467666) q[13];
ry(1.601723075417283e-05) q[14];
rz(-1.4846834196228964) q[14];
ry(0.0003970265357835581) q[15];
rz(2.427567322033575) q[15];
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
ry(-2.553129029969625e-06) q[0];
rz(-1.4269791550588338) q[0];
ry(-2.244962882098509) q[1];
rz(-2.5229929908545063) q[1];
ry(3.1415925571179923) q[2];
rz(0.20161865281154875) q[2];
ry(-0.000132299607504191) q[3];
rz(-1.2891923211388505) q[3];
ry(-3.141591196832842) q[4];
rz(1.4920464384423484) q[4];
ry(1.8644860214116017) q[5];
rz(-0.08100166478113113) q[5];
ry(-3.1374472026876608) q[6];
rz(2.7316623480864153) q[6];
ry(3.12630063673129) q[7];
rz(-1.8972442840081767) q[7];
ry(-9.398549218352968e-05) q[8];
rz(3.068884580039177) q[8];
ry(3.141386204105114) q[9];
rz(-2.3342390735870526) q[9];
ry(1.5710939602314502) q[10];
rz(-2.9543728871894115) q[10];
ry(3.1332701916612744) q[11];
rz(-0.9927545268845472) q[11];
ry(3.083494666815915) q[12];
rz(2.8600120455606257) q[12];
ry(-0.1928377748803669) q[13];
rz(2.7573093236548596) q[13];
ry(1.5641977944330598) q[14];
rz(-1.8201423996480315) q[14];
ry(-1.5697887842469964) q[15];
rz(0.00012375324133406172) q[15];
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
ry(0.2209455459440903) q[0];
rz(-1.2753633418481058) q[0];
ry(-0.3247856602926375) q[1];
rz(-0.5306919343196413) q[1];
ry(1.5707963798225295) q[2];
rz(3.1415819737103985) q[2];
ry(-1.722907234616619) q[3];
rz(2.6845394654970964) q[3];
ry(1.2355281153730857e-07) q[4];
rz(-3.0345743144668695) q[4];
ry(3.1306385921495616) q[5];
rz(-0.14005572891573073) q[5];
ry(0.0021862316639449375) q[6];
rz(1.308734184520711) q[6];
ry(-0.0032167172658157384) q[7];
rz(1.8001847958420285) q[7];
ry(2.7542904903784125e-05) q[8];
rz(-2.653146716739506) q[8];
ry(0.00019279092624894663) q[9];
rz(-3.017281333038799) q[9];
ry(9.140644539851193e-05) q[10];
rz(2.2636403870210775) q[10];
ry(-3.141478729077709) q[11];
rz(-2.7208032103093682) q[11];
ry(-0.0002519375698617131) q[12];
rz(1.7939393071707386) q[12];
ry(-0.00020246240314305763) q[13];
rz(-0.8730001885123091) q[13];
ry(-3.1415591253756467) q[14];
rz(-0.7148018541047803) q[14];
ry(-1.5699256801341717) q[15];
rz(2.1699933893232974) q[15];
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
ry(5.677492678390905e-05) q[0];
rz(-1.6548010432782023) q[0];
ry(1.2848381658663913e-05) q[1];
rz(2.971411073289214) q[1];
ry(-1.5707946833034665) q[2];
rz(-0.007332810516291059) q[2];
ry(-6.003770227636096e-07) q[3];
rz(-2.669543533131451) q[3];
ry(-1.5708542603986537) q[4];
rz(-0.6621481311496877) q[4];
ry(6.045186283909487e-05) q[5];
rz(0.054363615470725796) q[5];
ry(-9.65283529819061e-06) q[6];
rz(-2.5054880857055855) q[6];
ry(3.1415764793717265) q[7];
rz(-0.09055872895203264) q[7];
ry(3.141586375779606) q[8];
rz(0.6325530676857642) q[8];
ry(-3.141590014833963) q[9];
rz(-0.911272788617329) q[9];
ry(5.878923166058314e-06) q[10];
rz(-2.2711340725890827) q[10];
ry(3.1415800270765444) q[11];
rz(-2.3507668507986725) q[11];
ry(-3.141569385779227) q[12];
rz(-0.07765641615237247) q[12];
ry(-1.486910010672915e-05) q[13];
rz(-0.30324225145106914) q[13];
ry(3.1415925322335285) q[14];
rz(2.7306236107656394) q[14];
ry(2.3448100470006202e-05) q[15];
rz(2.5422922176392673) q[15];
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
ry(0.005140772985606348) q[0];
rz(-0.5012192784014768) q[0];
ry(3.138226685524595) q[1];
rz(-0.2480167606840533) q[1];
ry(-1.569472573732372) q[2];
rz(1.8574844649564575) q[2];
ry(0.9855583823539155) q[3];
rz(-0.9828181080841505) q[3];
ry(-2.7965986986622227e-05) q[4];
rz(-2.925536627451915) q[4];
ry(1.8089330284607597) q[5];
rz(-1.5880836908940774) q[5];
ry(1.8410374383077701) q[6];
rz(1.391686041264598) q[6];
ry(-1.3326199723347898) q[7];
rz(-1.5519236037528585) q[7];
ry(2.4421300772629926) q[8];
rz(-3.056295540028844) q[8];
ry(0.0009597081922425588) q[9];
rz(-0.36885879963232915) q[9];
ry(1.5630339238494853) q[10];
rz(1.5659052257253485) q[10];
ry(2.06063832561278) q[11];
rz(-0.912174710874595) q[11];
ry(-1.9998412817404736) q[12];
rz(1.6128512933081014) q[12];
ry(1.5708288164599562) q[13];
rz(-1.9959878575437493) q[13];
ry(0.002303455355991524) q[14];
rz(-1.6852153594472679) q[14];
ry(-1.5707274735189423) q[15];
rz(-1.4716732952492588) q[15];
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
ry(-3.141566613260446) q[0];
rz(-0.5463669581361519) q[0];
ry(-0.00015508591097290747) q[1];
rz(-1.5189371426038947) q[1];
ry(0.001765963251004184) q[2];
rz(0.025775649311714565) q[2];
ry(-3.1384687503772355) q[3];
rz(-1.4436298902164744) q[3];
ry(0.00035169628710196577) q[4];
rz(-1.1246763992586297) q[4];
ry(0.034040014593558425) q[5];
rz(0.04902228926914099) q[5];
ry(1.571018021981537) q[6];
rz(-2.8920207354138916) q[6];
ry(-2.8765738795080806) q[7];
rz(-1.852960191771001) q[7];
ry(1.5707802239592006) q[8];
rz(3.1415914053279645) q[8];
ry(-0.2638298074899296) q[9];
rz(0.5137282321128968) q[9];
ry(0.09423793149062387) q[10];
rz(1.8719595512850795) q[10];
ry(0.03718698255742435) q[11];
rz(1.313029526950725) q[11];
ry(-0.013055168595538902) q[12];
rz(2.597546759926908) q[12];
ry(-3.138111443743371) q[13];
rz(-2.076897740232506) q[13];
ry(3.1415809289721817) q[14];
rz(-2.7551252976456877) q[14];
ry(5.228027685788561e-05) q[15];
rz(-0.9740363351464651) q[15];
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
ry(4.7556633967664935e-06) q[0];
rz(3.0951936626672336) q[0];
ry(3.1415887523393797) q[1];
rz(-0.13942626305226644) q[1];
ry(2.876370034510245e-05) q[2];
rz(-2.022035854359614) q[2];
ry(-1.4567707910921388e-05) q[3];
rz(0.4289106083287561) q[3];
ry(1.570771648993198) q[4];
rz(-1.1070174474883714) q[4];
ry(-3.188467060620032e-06) q[5];
rz(-0.05153604971894114) q[5];
ry(3.141590584502223) q[6];
rz(2.256586584282452) q[6];
ry(6.848747151622092e-07) q[7];
rz(1.852726526184993) q[7];
ry(1.5708129618082018) q[8];
rz(-2.961343861824991) q[8];
ry(-1.81399480680966e-06) q[9];
rz(-1.3698087889473491) q[9];
ry(-3.1415921096057415) q[10];
rz(1.386264722849017) q[10];
ry(-1.2885042304233707e-06) q[11];
rz(2.3720760299840653) q[11];
ry(-3.6414728397105023e-06) q[12];
rz(0.5275272425769106) q[12];
ry(3.1415862874102958) q[13];
rz(0.24906541062198873) q[13];
ry(5.003229396267205e-06) q[14];
rz(-2.026269400433694) q[14];
ry(3.1415820304132316) q[15];
rz(-0.1890828343919093) q[15];
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
ry(-1.5835323669354269) q[0];
rz(-1.5717326711429551) q[0];
ry(1.531840131967561) q[1];
rz(1.570857797052807) q[1];
ry(-1.6727220827463134) q[2];
rz(1.570725761501985) q[2];
ry(1.3055132257719617) q[3];
rz(-1.561533355917069) q[3];
ry(-3.1297110586495105) q[4];
rz(-2.677837294825922) q[4];
ry(-0.26477357905702803) q[5];
rz(1.571400863687633) q[5];
ry(-0.11044531970769393) q[6];
rz(-2.009497210723499) q[6];
ry(3.106674843769779) q[7];
rz(-1.5715445038425324) q[7];
ry(-3.1379162037007378) q[8];
rz(1.7502896263883265) q[8];
ry(-3.1358180653359495) q[9];
rz(0.7140583096803769) q[9];
ry(-3.1142646340514832) q[10];
rz(1.0899175152543106) q[10];
ry(3.139797054079694) q[11];
rz(2.114959417341119) q[11];
ry(-3.087449634435876) q[12];
rz(-1.5584066372048655) q[12];
ry(4.756335518716357e-05) q[13];
rz(-0.3344863961258264) q[13];
ry(2.4890673184134844) q[14];
rz(-1.5786019878706528) q[14];
ry(0.17730253403090668) q[15];
rz(1.5626074925176772) q[15];
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
ry(1.5707971782406265) q[0];
rz(3.1415611865964266) q[0];
ry(-1.5707977338595658) q[1];
rz(-3.14152308212716) q[1];
ry(1.570794620920509) q[2];
rz(-0.00013798069407260777) q[2];
ry(1.5708018373614485) q[3];
rz(0.001302652145852681) q[3];
ry(-1.5714107951034844) q[4];
rz(1.5764818344772618) q[4];
ry(1.5707918563040948) q[5];
rz(-3.140290470146811) q[5];
ry(1.5707991202938025) q[6];
rz(3.1414523943513775) q[6];
ry(1.5707996564518023) q[7];
rz(6.158239700493338e-05) q[7];
ry(1.5708023357056726) q[8];
rz(3.1415870093119773) q[8];
ry(1.5708019834950795) q[9];
rz(8.611276349768104e-06) q[9];
ry(1.5708013972283437) q[10];
rz(3.141591131828438) q[10];
ry(-1.5707933836448484) q[11];
rz(-1.662024988509426e-07) q[11];
ry(-1.5707937466343143) q[12];
rz(3.141592564037083) q[12];
ry(-1.570795720128878) q[13];
rz(-1.3212155884789441e-07) q[13];
ry(1.5707970722055717) q[14];
rz(3.1415921417448427) q[14];
ry(1.5707967214838634) q[15];
rz(-4.0189224172559064e-07) q[15];