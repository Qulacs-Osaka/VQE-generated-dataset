OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-2.055717858182067) q[0];
rz(0.2772701133322304) q[0];
ry(1.0422766052742665) q[1];
rz(-0.12385200134766736) q[1];
ry(0.21045163739959727) q[2];
rz(-2.336781330239437) q[2];
ry(1.8056544288522893) q[3];
rz(-0.6182308780258554) q[3];
ry(-2.304061105423946) q[4];
rz(0.47164612531600003) q[4];
ry(0.9649925966821195) q[5];
rz(-0.9575150225613385) q[5];
ry(3.0937694076203277) q[6];
rz(-2.1101529017412415) q[6];
ry(-2.1712900089403693) q[7];
rz(2.0663233763501188) q[7];
ry(2.2387620694939723) q[8];
rz(1.1304069310579705) q[8];
ry(-3.055808323781112) q[9];
rz(3.0698325123527277) q[9];
ry(-1.3353201929074414) q[10];
rz(2.37321240597122) q[10];
ry(-1.7597462013651877) q[11];
rz(-2.845744640574558) q[11];
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
ry(0.768879157664325) q[0];
rz(-1.26585494482744) q[0];
ry(1.26638515745166) q[1];
rz(-0.2941563743292015) q[1];
ry(-2.314290659744132) q[2];
rz(-2.0095762283348435) q[2];
ry(1.8777224587993722) q[3];
rz(0.08979723692817353) q[3];
ry(1.279396988524243) q[4];
rz(-0.5878078042041635) q[4];
ry(1.42267342381135) q[5];
rz(0.6864338238391287) q[5];
ry(2.7882869065821514) q[6];
rz(-1.034499304120768) q[6];
ry(2.9089538940382753) q[7];
rz(-3.042621813590689) q[7];
ry(0.7732155398734301) q[8];
rz(-0.5870665631236651) q[8];
ry(1.1208195799039844) q[9];
rz(-1.4662776701876334) q[9];
ry(2.271169082444432) q[10];
rz(2.064468008245633) q[10];
ry(-0.527060657860571) q[11];
rz(0.7109083929482737) q[11];
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
ry(2.8107145419438155) q[0];
rz(1.4130845570720465) q[0];
ry(1.8144855214166529) q[1];
rz(0.889075926838222) q[1];
ry(-2.101013437358695) q[2];
rz(1.1166469088199373) q[2];
ry(0.9958086619039876) q[3];
rz(0.04884351435840184) q[3];
ry(-2.381533514247325) q[4];
rz(2.331359874904356) q[4];
ry(-1.8093413383370536) q[5];
rz(1.6803778147083375) q[5];
ry(1.8607752833729352) q[6];
rz(-2.458445864063721) q[6];
ry(-1.357390480892561) q[7];
rz(1.1370113432450388) q[7];
ry(1.138723861680754) q[8];
rz(-2.0334258814372586) q[8];
ry(1.5066535898747675) q[9];
rz(2.512300100402753) q[9];
ry(0.7020739685354983) q[10];
rz(1.4174836551533758) q[10];
ry(2.3239918019547847) q[11];
rz(-0.5332812866243509) q[11];
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
ry(-2.296611142184304) q[0];
rz(0.8031542843773443) q[0];
ry(-0.6204677963282562) q[1];
rz(-1.0231107967413082) q[1];
ry(0.8611469582996651) q[2];
rz(-0.3152773368147681) q[2];
ry(-0.6736109627264639) q[3];
rz(1.4372075127437318) q[3];
ry(2.789941389382737) q[4];
rz(0.24213315418660755) q[4];
ry(-2.457525886072281) q[5];
rz(-0.5854221891265512) q[5];
ry(-2.6123611253395964) q[6];
rz(0.7386936197067052) q[6];
ry(0.33787407000475805) q[7];
rz(-2.6608460088064327) q[7];
ry(1.3921016944712035) q[8];
rz(-1.0205158906273673) q[8];
ry(1.8104200533745827) q[9];
rz(2.4482662255600647) q[9];
ry(0.8250961517239892) q[10];
rz(-1.4786125517175277) q[10];
ry(1.7874475387127415) q[11];
rz(-1.22525286377198) q[11];
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
ry(1.3223146861029518) q[0];
rz(2.001307264105988) q[0];
ry(1.2072610781617374) q[1];
rz(-1.2322964855954197) q[1];
ry(1.0920199208602943) q[2];
rz(1.8702274510932682) q[2];
ry(0.2820029438706167) q[3];
rz(-2.8630974431010836) q[3];
ry(-0.42997291187213804) q[4];
rz(0.8229513085196817) q[4];
ry(-1.942671430112607) q[5];
rz(-0.7620143663982724) q[5];
ry(-1.5831617884259268) q[6];
rz(2.32926526018736) q[6];
ry(-2.676908020660752) q[7];
rz(0.168881614762423) q[7];
ry(1.0340144132685904) q[8];
rz(1.4743364341185106) q[8];
ry(3.024995211409546) q[9];
rz(0.5619876516752532) q[9];
ry(2.5864788783374606) q[10];
rz(-0.5016677615871561) q[10];
ry(-2.893575282417941) q[11];
rz(0.7656754700811783) q[11];
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
ry(1.4145630340480404) q[0];
rz(-1.5634154943505232) q[0];
ry(-1.4590701093061806) q[1];
rz(-2.8684820958608883) q[1];
ry(-2.7508543283774682) q[2];
rz(-0.4528539589124722) q[2];
ry(-2.0135850640330375) q[3];
rz(-2.0660901661789057) q[3];
ry(-1.2419767196369975) q[4];
rz(0.9809254704475827) q[4];
ry(-0.45649761991407) q[5];
rz(2.015082340568247) q[5];
ry(-2.8521349418295143) q[6];
rz(0.7476243309201889) q[6];
ry(-2.9338882428157316) q[7];
rz(0.8957598442751371) q[7];
ry(0.7357937528565676) q[8];
rz(-1.4923553299363004) q[8];
ry(-1.3926007069053052) q[9];
rz(-1.4004860282864602) q[9];
ry(2.5724908262354327) q[10];
rz(-0.29555141888910497) q[10];
ry(2.605758667804281) q[11];
rz(-0.6025627235919124) q[11];
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
ry(1.5993369009694007) q[0];
rz(2.6425537495501947) q[0];
ry(-2.808786007459441) q[1];
rz(1.33073032199996) q[1];
ry(1.010724220051551) q[2];
rz(3.071020103750687) q[2];
ry(-2.4707103106592943) q[3];
rz(-1.3754263705174337) q[3];
ry(1.0974901944848852) q[4];
rz(-0.9057471805384769) q[4];
ry(0.490018640448386) q[5];
rz(-2.9277725339628105) q[5];
ry(2.402147743726482) q[6];
rz(1.3183430645535985) q[6];
ry(-0.5931910100369427) q[7];
rz(-1.672753944661695) q[7];
ry(-2.403546542097519) q[8];
rz(0.5981337140727632) q[8];
ry(1.8519598804782729) q[9];
rz(3.037429088840855) q[9];
ry(1.4075120153655354) q[10];
rz(-2.539323910170721) q[10];
ry(-0.6914603151812543) q[11];
rz(1.3359284481446245) q[11];
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
ry(1.9448922231603012) q[0];
rz(2.150613032095567) q[0];
ry(2.525753893701784) q[1];
rz(-1.0996104454388362) q[1];
ry(-0.4021494451548232) q[2];
rz(1.6823659604772099) q[2];
ry(1.4994407377816774) q[3];
rz(-0.07578392104502887) q[3];
ry(-1.7431431498688617) q[4];
rz(-1.1358501311013036) q[4];
ry(2.132865853770319) q[5];
rz(1.1702374871363075) q[5];
ry(-2.585323998788802) q[6];
rz(-3.033095537576266) q[6];
ry(2.9584775298563697) q[7];
rz(1.7754825128951774) q[7];
ry(-3.0904214945709287) q[8];
rz(0.3807585430442604) q[8];
ry(-2.7785154655169912) q[9];
rz(-1.7952001003405647) q[9];
ry(1.8669390859157375) q[10];
rz(0.5940461374203382) q[10];
ry(-2.7657017129315893) q[11];
rz(-2.347947126278891) q[11];
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
ry(-0.2722047684290838) q[0];
rz(-1.1866258488526595) q[0];
ry(-0.5819812674255497) q[1];
rz(-2.273246664507044) q[1];
ry(-1.339325624330803) q[2];
rz(0.5401106612044653) q[2];
ry(-2.28761016316763) q[3];
rz(1.371671916737759) q[3];
ry(-1.7635275743976253) q[4];
rz(2.9801968684649665) q[4];
ry(-1.3889724506770371) q[5];
rz(2.318573989683619) q[5];
ry(-2.620739176173014) q[6];
rz(1.423862803704995) q[6];
ry(0.2087369890197781) q[7];
rz(1.7945189886936725) q[7];
ry(0.10686495680192787) q[8];
rz(-0.236880030272232) q[8];
ry(-2.6010781543224506) q[9];
rz(0.49366905626407803) q[9];
ry(-0.16854157502094846) q[10];
rz(2.0502554802504074) q[10];
ry(-0.6505813042450797) q[11];
rz(1.0263305848376323) q[11];
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
ry(2.263005413183637) q[0];
rz(0.15170159055872381) q[0];
ry(2.6765921709341134) q[1];
rz(-2.926745411953564) q[1];
ry(-1.0591847922321291) q[2];
rz(-0.16808457047374445) q[2];
ry(-0.8208290176828648) q[3];
rz(-2.174518299550606) q[3];
ry(1.5641389497873233) q[4];
rz(1.6482991503527957) q[4];
ry(-0.6776327996055136) q[5];
rz(0.6190793767680841) q[5];
ry(1.556500720834463) q[6];
rz(1.5506868795440816) q[6];
ry(1.903203010406666) q[7];
rz(-1.4334861881709466) q[7];
ry(-0.3922421167583874) q[8];
rz(1.5113756583182076) q[8];
ry(-2.6130356038844997) q[9];
rz(-0.2576510711305122) q[9];
ry(-0.8097449751810418) q[10];
rz(-2.678893160431199) q[10];
ry(2.141105540415557) q[11];
rz(2.086753761084352) q[11];
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
ry(0.32375603855750157) q[0];
rz(-1.7867252764585515) q[0];
ry(-2.585565442479027) q[1];
rz(1.6082921757827224) q[1];
ry(1.1502590791462985) q[2];
rz(2.983162917992297) q[2];
ry(1.3502201038849613) q[3];
rz(-2.8314517664097014) q[3];
ry(0.2587446025418998) q[4];
rz(1.4795070229116125) q[4];
ry(-1.3076440198934298) q[5];
rz(-0.3411980187046266) q[5];
ry(2.213702792985638) q[6];
rz(0.19235207794855264) q[6];
ry(0.0856633425832305) q[7];
rz(-1.2510964463853718) q[7];
ry(-2.5598008772766745) q[8];
rz(0.8301234731712289) q[8];
ry(0.6327308998318601) q[9];
rz(-2.910912272046963) q[9];
ry(1.248014445502303) q[10];
rz(-1.0375949676437968) q[10];
ry(2.490132084087388) q[11];
rz(1.964163618161995) q[11];
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
ry(-0.3257836711780211) q[0];
rz(-1.7773539554382198) q[0];
ry(1.9098719833639448) q[1];
rz(2.3021327326271397) q[1];
ry(2.9153267425782055) q[2];
rz(-2.6445280062596646) q[2];
ry(-1.941526637154805) q[3];
rz(0.8139793248880975) q[3];
ry(-1.6877783828978978) q[4];
rz(0.8661160233206645) q[4];
ry(-2.824507120380772) q[5];
rz(2.9927593344948624) q[5];
ry(2.212683599258437) q[6];
rz(-1.2363071798810408) q[6];
ry(-0.5718393774658619) q[7];
rz(1.6413287841252249) q[7];
ry(0.33796109509898553) q[8];
rz(0.2531950495880802) q[8];
ry(2.6175157116953782) q[9];
rz(0.8245293278918582) q[9];
ry(1.3421682820912277) q[10];
rz(-1.6516308573160063) q[10];
ry(-1.4278905275620895) q[11];
rz(-0.31901509526053934) q[11];
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
ry(-0.2851496862204429) q[0];
rz(-0.5373724953560348) q[0];
ry(1.1504851497726265) q[1];
rz(-0.7277216554986523) q[1];
ry(-1.1782912223638642) q[2];
rz(1.8679418791851692) q[2];
ry(2.4415466393563503) q[3];
rz(1.769352544938881) q[3];
ry(2.8181469065223994) q[4];
rz(1.311017819203017) q[4];
ry(1.0632737878429488) q[5];
rz(1.6391395938509021) q[5];
ry(-0.8920175254791463) q[6];
rz(1.609408156995355) q[6];
ry(-2.207247114439667) q[7];
rz(1.6101522688435277) q[7];
ry(-0.39154768005648455) q[8];
rz(-0.8679225165993243) q[8];
ry(1.1827569697012523) q[9];
rz(-1.1057749874964093) q[9];
ry(-3.012767216527415) q[10];
rz(1.3507923579624181) q[10];
ry(-0.5256947531051299) q[11];
rz(1.7905888822865972) q[11];
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
ry(2.1391726000578903) q[0];
rz(1.0574710452883478) q[0];
ry(-0.44859467590335594) q[1];
rz(-0.46051170359505716) q[1];
ry(-2.4584775759252846) q[2];
rz(2.3406665738117165) q[2];
ry(2.155839831448736) q[3];
rz(-2.4575963941405) q[3];
ry(-1.602364404776755) q[4];
rz(-2.665536004877039) q[4];
ry(0.30375894899158684) q[5];
rz(0.7635550966152813) q[5];
ry(2.413329514597303) q[6];
rz(-1.2414063788165848) q[6];
ry(-1.4710356260171553) q[7];
rz(-2.7876864387645464) q[7];
ry(2.4900752913784525) q[8];
rz(-2.3615924034988742) q[8];
ry(2.252946808711237) q[9];
rz(0.3770136830465897) q[9];
ry(1.8569977100846309) q[10];
rz(0.5663174765759126) q[10];
ry(-1.2938224214756624) q[11];
rz(-1.861212109303473) q[11];
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
ry(-1.5416198432708668) q[0];
rz(-1.8132539723954222) q[0];
ry(1.090552141666043) q[1];
rz(-2.711789378384906) q[1];
ry(0.5420589565623297) q[2];
rz(-0.03510824852510186) q[2];
ry(2.7922931899638312) q[3];
rz(-0.1042706964928391) q[3];
ry(2.8047488977933823) q[4];
rz(1.5268007950167846) q[4];
ry(-3.1015161848084163) q[5];
rz(-2.2060526986578806) q[5];
ry(-1.18795878543584) q[6];
rz(-0.45573403368752624) q[6];
ry(2.9729574877750577) q[7];
rz(-1.1919783118806315) q[7];
ry(1.7235973258410242) q[8];
rz(-0.7355659874025057) q[8];
ry(-1.2071026419231288) q[9];
rz(2.1313767216853314) q[9];
ry(-2.451936807799801) q[10];
rz(2.91577365123967) q[10];
ry(-1.7997923889906482) q[11];
rz(-1.1053911054193053) q[11];
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
ry(1.4125334608615363) q[0];
rz(2.1831278361236555) q[0];
ry(2.6675735049564975) q[1];
rz(-0.9050688266763022) q[1];
ry(1.3733782398036212) q[2];
rz(-2.129678799489693) q[2];
ry(-2.795436448815882) q[3];
rz(2.837599543606926) q[3];
ry(-1.0135082103545727) q[4];
rz(-2.7536952885833643) q[4];
ry(2.0771045304286364) q[5];
rz(-1.5333062939213902) q[5];
ry(-1.9012217186810698) q[6];
rz(-0.2382852131749251) q[6];
ry(-2.940390645845126) q[7];
rz(2.8016060581316586) q[7];
ry(-2.700498759687377) q[8];
rz(0.9679565224487305) q[8];
ry(2.9730838959916004) q[9];
rz(2.5767468508839824) q[9];
ry(1.5304025577328408) q[10];
rz(1.764874664387162) q[10];
ry(0.7053988054800895) q[11];
rz(-2.118654946923839) q[11];
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
ry(-2.2229259059027973) q[0];
rz(-2.5642708787388644) q[0];
ry(2.549272684294702) q[1];
rz(-2.352280578392592) q[1];
ry(2.51848819910019) q[2];
rz(1.3337289717911547) q[2];
ry(-1.4913327552884095) q[3];
rz(-0.3569376125929545) q[3];
ry(-2.502763501519458) q[4];
rz(0.030100100408645372) q[4];
ry(-0.6694622563756313) q[5];
rz(-0.30668910954694606) q[5];
ry(2.5135143345651505) q[6];
rz(0.36588990321244) q[6];
ry(0.5928950521593803) q[7];
rz(-1.1345400234452012) q[7];
ry(3.0120429776147204) q[8];
rz(1.1296893288583192) q[8];
ry(0.4798500518000088) q[9];
rz(-0.5804359813704316) q[9];
ry(-3.085509400987431) q[10];
rz(-0.29181736707381756) q[10];
ry(-2.3463256859682673) q[11];
rz(0.1995221620055911) q[11];
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
ry(2.6621042630425036) q[0];
rz(-0.8710548895976932) q[0];
ry(1.6045743770375012) q[1];
rz(-0.9493102010741712) q[1];
ry(-1.7698047605828986) q[2];
rz(-0.5587013407918162) q[2];
ry(-1.785331427906457) q[3];
rz(-2.0866577415836067) q[3];
ry(-2.6521248987916817) q[4];
rz(-0.813640012026128) q[4];
ry(-1.9657733218844795) q[5];
rz(1.4880165701248613) q[5];
ry(2.7260133909576703) q[6];
rz(-1.8648385225480528) q[6];
ry(1.551840866518623) q[7];
rz(-1.5435851772784763) q[7];
ry(-0.8343777448719126) q[8];
rz(2.5863232262464195) q[8];
ry(-2.6167888119565292) q[9];
rz(-1.8715190106867974) q[9];
ry(2.1865722966448664) q[10];
rz(-2.0321098148010925) q[10];
ry(-2.508392479342528) q[11];
rz(-1.9966703589365506) q[11];
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
ry(2.2353722479612848) q[0];
rz(-1.7941688370203597) q[0];
ry(1.9541178547110665) q[1];
rz(0.372333339206487) q[1];
ry(0.15619138091217177) q[2];
rz(0.9548058092457588) q[2];
ry(1.2033367999250097) q[3];
rz(2.019103180236635) q[3];
ry(-1.446168262292197) q[4];
rz(-2.2613924398223446) q[4];
ry(0.9577500365516292) q[5];
rz(-0.4829854454453439) q[5];
ry(-0.6408298199980648) q[6];
rz(2.626850326295685) q[6];
ry(1.5355903043968147) q[7];
rz(-2.8917189761711644) q[7];
ry(0.770551129046133) q[8];
rz(0.7879185182035435) q[8];
ry(-2.027994530777717) q[9];
rz(-2.1902882466973415) q[9];
ry(2.1227203112584023) q[10];
rz(2.136911666640998) q[10];
ry(-2.0086250303332083) q[11];
rz(1.8835898613839) q[11];
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
ry(1.0791307098202703) q[0];
rz(0.8193692850648979) q[0];
ry(2.5287504477476532) q[1];
rz(-2.4300396403882942) q[1];
ry(-1.4349160449183889) q[2];
rz(-0.17412970788003987) q[2];
ry(0.32273177566716177) q[3];
rz(2.6450377913201444) q[3];
ry(2.1101176034320774) q[4];
rz(2.6048434467740376) q[4];
ry(-1.7692595697994173) q[5];
rz(-2.0315210960423715) q[5];
ry(0.06693118287145516) q[6];
rz(2.9553738135651035) q[6];
ry(0.9290261315566265) q[7];
rz(-2.8349380750548026) q[7];
ry(0.7507527697119336) q[8];
rz(-0.7869302761889473) q[8];
ry(-1.4922616867447474) q[9];
rz(-2.1915176961334093) q[9];
ry(-1.951267604917561) q[10];
rz(0.21835000411132005) q[10];
ry(-2.516849402714538) q[11];
rz(-1.277308696291362) q[11];
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
ry(-1.574699284103574) q[0];
rz(-1.7641536514704712) q[0];
ry(-0.17092543947607788) q[1];
rz(-1.1137651575234349) q[1];
ry(1.233038765061866) q[2];
rz(1.1939399847236174) q[2];
ry(0.3567925668552947) q[3];
rz(-0.876268109108661) q[3];
ry(2.180351155040368) q[4];
rz(-1.2876554578552497) q[4];
ry(0.9406720018638639) q[5];
rz(-1.3869268616331114) q[5];
ry(-2.706393182035985) q[6];
rz(0.15488997825496725) q[6];
ry(0.06550324711794563) q[7];
rz(1.7006905368874525) q[7];
ry(1.5744336569054491) q[8];
rz(1.7082531148802473) q[8];
ry(2.689388084650782) q[9];
rz(-1.3454347194955103) q[9];
ry(1.6283669256935214) q[10];
rz(-2.2153937812008953) q[10];
ry(-1.3075435312190884) q[11];
rz(1.0267941659782542) q[11];
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
ry(-1.8283565347493322) q[0];
rz(-2.625999847397674) q[0];
ry(0.465148305598901) q[1];
rz(0.4743745361266976) q[1];
ry(-0.965514567421096) q[2];
rz(-0.400375481811209) q[2];
ry(-1.0343250645572883) q[3];
rz(-2.593134028293663) q[3];
ry(1.3713859967091313) q[4];
rz(-2.8815997965982403) q[4];
ry(0.7251876210992316) q[5];
rz(-1.2469206400672839) q[5];
ry(-2.364256998025843) q[6];
rz(-0.7950579454630847) q[6];
ry(1.5440749869364592) q[7];
rz(-0.4976882807069049) q[7];
ry(-1.2482836129544435) q[8];
rz(-1.8657480171656182) q[8];
ry(0.6213757196382188) q[9];
rz(0.4280117900394211) q[9];
ry(1.6768697524407916) q[10];
rz(-2.5593040986452027) q[10];
ry(1.076663600212946) q[11];
rz(-1.372519886294869) q[11];
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
ry(-2.0823961037981062) q[0];
rz(2.3445895619220853) q[0];
ry(1.4645538667386315) q[1];
rz(1.6293524278981686) q[1];
ry(-2.179058922090649) q[2];
rz(0.5350641641902678) q[2];
ry(-0.9976882921871334) q[3];
rz(0.6593434739802786) q[3];
ry(0.152958440529045) q[4];
rz(-2.3941521170867843) q[4];
ry(-1.9494901257902169) q[5];
rz(-2.46933639269728) q[5];
ry(1.009777573159119) q[6];
rz(0.25528476585495863) q[6];
ry(-0.10512800791857746) q[7];
rz(1.1354934082389083) q[7];
ry(-2.4908172509558737) q[8];
rz(-2.7047411714594647) q[8];
ry(-1.5471975306281556) q[9];
rz(1.4903024116384158) q[9];
ry(-2.763875634759125) q[10];
rz(-1.369427316113696) q[10];
ry(-0.4397688860195075) q[11];
rz(-0.7533898209813614) q[11];
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
ry(0.9918830015978735) q[0];
rz(-0.7644697358882943) q[0];
ry(-2.4029314372381596) q[1];
rz(0.06297535616425791) q[1];
ry(-0.9323534898162068) q[2];
rz(1.9402024836744751) q[2];
ry(2.516510843744966) q[3];
rz(1.7215924289261881) q[3];
ry(0.42393149166436306) q[4];
rz(-0.8314878381570612) q[4];
ry(2.5592627047845498) q[5];
rz(-2.6563613593141833) q[5];
ry(2.7181342359478595) q[6];
rz(-2.588105432575458) q[6];
ry(2.28965542479664) q[7];
rz(1.7203686920026502) q[7];
ry(1.4267022976701007) q[8];
rz(0.7886325597993391) q[8];
ry(2.254881156025447) q[9];
rz(-3.0344854353189876) q[9];
ry(-0.761802196350299) q[10];
rz(1.4345124329760317) q[10];
ry(-1.8988315808198313) q[11];
rz(-0.2727985721000641) q[11];