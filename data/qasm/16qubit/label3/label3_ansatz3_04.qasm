OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.45564532333771) q[0];
rz(0.0005741793912155743) q[0];
ry(-1.5696958257990694) q[1];
rz(1.810570928002466) q[1];
ry(0.009544769579253164) q[2];
rz(-1.797506692992294) q[2];
ry(1.569505576800056) q[3];
rz(-0.7283762584760547) q[3];
ry(-1.6079872196735057) q[4];
rz(1.7136341124247336) q[4];
ry(-3.1346507868868283) q[5];
rz(-2.8475662853274106) q[5];
ry(2.9687238643688887) q[6];
rz(-1.4831756738540687) q[6];
ry(-3.136799550420937) q[7];
rz(1.4459912261384202) q[7];
ry(-0.002916956809701776) q[8];
rz(-1.687624945243738) q[8];
ry(-1.8641080300672497) q[9];
rz(1.6534317300110455) q[9];
ry(-2.777571553679212) q[10];
rz(-2.6782184793816954) q[10];
ry(-1.5393028505563502) q[11];
rz(3.0018673992125264) q[11];
ry(-3.1408996112489285) q[12];
rz(0.6238811596953031) q[12];
ry(-3.1413093562981342) q[13];
rz(-0.005335602609848777) q[13];
ry(2.7081989457658397) q[14];
rz(-1.359288479279496) q[14];
ry(2.9224893181631164) q[15];
rz(-2.5579869677713765) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.5728221174201316) q[0];
rz(-0.5331019200224674) q[0];
ry(-2.8516830250817566) q[1];
rz(1.2816392422639256) q[1];
ry(1.5710772227834522) q[2];
rz(1.909288878432553) q[2];
ry(2.083674636179661) q[3];
rz(-1.0987714935179707) q[3];
ry(2.885727578999038) q[4];
rz(-0.5483830451705778) q[4];
ry(1.5699268230716559) q[5];
rz(3.0623773720544554) q[5];
ry(-0.01241613637828101) q[6];
rz(-2.634040058494623) q[6];
ry(0.012229055165034808) q[7];
rz(2.368496187792717) q[7];
ry(-0.003636888929576365) q[8];
rz(-1.1152691690872854) q[8];
ry(0.1424060555286415) q[9];
rz(1.488586459873086) q[9];
ry(1.5617357737347985) q[10];
rz(-1.5802130319944077) q[10];
ry(3.0257396744047824) q[11];
rz(-0.002739994149443408) q[11];
ry(1.5710668294223726) q[12];
rz(3.136649448563868) q[12];
ry(0.8456346679903441) q[13];
rz(1.9135793119295128) q[13];
ry(-2.1997056469719087) q[14];
rz(-1.099612004829098) q[14];
ry(-1.8080725770554829) q[15];
rz(1.59850394215417) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(0.00019250370320733626) q[0];
rz(-1.04455948645116) q[0];
ry(1.7163369243503792) q[1];
rz(-0.04129378356478775) q[1];
ry(2.873545695230839) q[2];
rz(1.8257956885246778) q[2];
ry(1.2538183308308894) q[3];
rz(1.607956109136788) q[3];
ry(-1.5740340490137106) q[4];
rz(-1.5730105838288815) q[4];
ry(3.0979374120573984) q[5];
rz(-0.11938809711197403) q[5];
ry(-3.092039053303841) q[6];
rz(0.05311222602003962) q[6];
ry(2.551816951980763) q[7];
rz(-1.7972358360468015) q[7];
ry(-3.131847446005635) q[8];
rz(2.008867484282162) q[8];
ry(1.8630965834456998) q[9];
rz(-2.6253381319702673) q[9];
ry(-0.07916015171546587) q[10];
rz(-1.5665889631583134) q[10];
ry(-0.3310841723251884) q[11];
rz(1.5697054173840979) q[11];
ry(0.05444696487750278) q[12];
rz(-1.5647625648426597) q[12];
ry(-2.1186641656113974) q[13];
rz(-1.5423352429000792) q[13];
ry(3.140279757796504) q[14];
rz(-0.07975524506866767) q[14];
ry(1.4332496308599423) q[15];
rz(1.5216077880987409) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(3.139530289175966) q[0];
rz(0.34086504685713326) q[0];
ry(1.5807524877171786) q[1];
rz(1.843728513469932) q[1];
ry(-1.570824351674922) q[2];
rz(1.5719648471617624) q[2];
ry(0.07388637225457835) q[3];
rz(2.4324565647692657) q[3];
ry(-1.5700365610478326) q[4];
rz(1.5716780850438155) q[4];
ry(3.117087436655054) q[5];
rz(-1.6119804422561788) q[5];
ry(-1.5691914575710175) q[6];
rz(-0.02459013379527697) q[6];
ry(3.134931184356786) q[7];
rz(-1.3823589657286923) q[7];
ry(-1.570777287287659) q[8];
rz(3.138211729085618) q[8];
ry(0.17077345331321367) q[9];
rz(2.895342389644748) q[9];
ry(-1.5725069363471285) q[10];
rz(-0.8058368028370325) q[10];
ry(1.887131537915728) q[11];
rz(1.570571587888806) q[11];
ry(-0.95108420833647) q[12];
rz(1.5702487307116382) q[12];
ry(2.031982501190648) q[13];
rz(1.2305551640070735) q[13];
ry(0.6373187054448635) q[14];
rz(2.455607020534074) q[14];
ry(1.5093534100237194) q[15];
rz(-2.938584673609046) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(0.5106707121940008) q[0];
rz(1.5724244398155363) q[0];
ry(-3.1387757573227844) q[1];
rz(-1.2879566001322855) q[1];
ry(-0.14033666127470087) q[2];
rz(-0.0006378491449936519) q[2];
ry(-3.141096317559086) q[3];
rz(2.5166334211787396) q[3];
ry(1.760223655218315) q[4];
rz(0.4254977415971659) q[4];
ry(1.5247482479647345) q[5];
rz(1.5734006300822694) q[5];
ry(3.119075158865431) q[6];
rz(1.2722380584116362) q[6];
ry(3.1407559954352213) q[7];
rz(1.182556810657415) q[7];
ry(2.6957692907072173) q[8];
rz(-0.0016459527562284746) q[8];
ry(0.011115496594240111) q[9];
rz(-0.8493249166327681) q[9];
ry(2.7022878363600684) q[10];
rz(2.9828070529539934) q[10];
ry(-1.5713045080730135) q[11];
rz(0.01836912486513302) q[11];
ry(1.5711103520589256) q[12];
rz(-0.5987506454863336) q[12];
ry(-0.191019320258546) q[13];
rz(2.614425734763474) q[13];
ry(-1.5710702959807878) q[14];
rz(1.5687732723043255) q[14];
ry(1.4479323136845899) q[15];
rz(-1.6537228236308295) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.571846784697844) q[0];
rz(0.00374787991932001) q[0];
ry(1.5706661057664606) q[1];
rz(2.65817140677138) q[1];
ry(2.974918672653465) q[2];
rz(-1.556261106659726) q[2];
ry(0.2697667470354398) q[3];
rz(-0.042512672229801034) q[3];
ry(3.1404197892183223) q[4];
rz(0.4258183120343981) q[4];
ry(2.859020697846007) q[5];
rz(1.5812399556097274) q[5];
ry(-3.141127805203712) q[6];
rz(-1.0562906174868507) q[6];
ry(0.0011987079459432899) q[7];
rz(1.8123131264289638) q[7];
ry(-1.5699315864988632) q[8];
rz(1.8367954290373703) q[8];
ry(-0.0017021231062512854) q[9];
rz(-2.1899087934648933) q[9];
ry(1.572616188534468) q[10];
rz(0.012749229201894228) q[10];
ry(-1.571127611090992) q[11];
rz(-3.1344927807104543) q[11];
ry(2.4036919194278012) q[12];
rz(-1.8298225930163556) q[12];
ry(-2.2484924956050647) q[13];
rz(-2.5674950246713113) q[13];
ry(0.1551398576739036) q[14];
rz(-3.139115770271201) q[14];
ry(-1.5711863030172388) q[15];
rz(-1.570083979636264) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-2.8623941073096644) q[0];
rz(-2.8434880621011454) q[0];
ry(-1.5689757469489982) q[1];
rz(-2.729355816666996) q[1];
ry(0.028347734415283377) q[2];
rz(3.128902240461563) q[2];
ry(1.5710789179542362) q[3];
rz(-1.5724455055985258) q[3];
ry(-1.3750452823706436) q[4];
rz(2.7920842765393346) q[4];
ry(2.789615707547262) q[5];
rz(0.010051183289879974) q[5];
ry(-3.1415364218895743) q[6];
rz(0.786079702045597) q[6];
ry(-0.0006760391964579652) q[7];
rz(2.3325445919974412) q[7];
ry(0.00022168726050482358) q[8];
rz(3.0902934047988335) q[8];
ry(0.2570759217518699) q[9];
rz(-1.908667977725802) q[9];
ry(0.003929783098052475) q[10];
rz(-0.015990374762866466) q[10];
ry(-3.0506289207638226) q[11];
rz(0.004888405205165647) q[11];
ry(-0.3503390190022813) q[12];
rz(-1.7775290381249014) q[12];
ry(-1.646806966953398) q[13];
rz(1.8150915288489937) q[13];
ry(-1.5713607284318192) q[14];
rz(2.6554000481418436) q[14];
ry(-1.5696150365157173) q[15];
rz(2.5817768610501592) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-3.140305289004272) q[0];
rz(0.44659682472653367) q[0];
ry(3.14066520298421) q[1];
rz(2.135654299417915) q[1];
ry(1.56342733359171) q[2];
rz(0.15171280348838925) q[2];
ry(-1.5685724242251418) q[3];
rz(0.1487743421277701) q[3];
ry(-0.0003121809613659853) q[4];
rz(0.5010700275441522) q[4];
ry(1.571445597695866) q[5];
rz(0.148802759387324) q[5];
ry(1.5700160498725388) q[6];
rz(-2.990078602096751) q[6];
ry(-0.001375168368941544) q[7];
rz(1.5228092359487917) q[7];
ry(-0.004813685740531426) q[8];
rz(1.5078545140832296) q[8];
ry(3.1405206603557203) q[9];
rz(1.337575501964822) q[9];
ry(-1.5702581513536087) q[10];
rz(0.0036458729890203045) q[10];
ry(1.5705068723354412) q[11];
rz(-2.9768470244624905) q[11];
ry(-3.1375746214964804) q[12];
rz(-2.1064842592551614) q[12];
ry(0.002138095338421131) q[13];
rz(-1.637022009424796) q[13];
ry(3.138076620281442) q[14];
rz(1.1357657410197408) q[14];
ry(-0.0027968639127173844) q[15];
rz(1.7142422650026203) q[15];