OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.5707174480383788) q[0];
rz(-0.01489637826428188) q[0];
ry(-1.5708222195169768) q[1];
rz(-1.6485105565641185) q[1];
ry(4.010027470702226e-05) q[2];
rz(0.7049538496047347) q[2];
ry(-1.570854238376012) q[3];
rz(-0.7978816976255663) q[3];
ry(1.5720133285375157) q[4];
rz(1.5488197735998224) q[4];
ry(0.0022553640235569716) q[5];
rz(1.5337912023681135) q[5];
ry(-1.5707707473533026) q[6];
rz(-2.5425352847530767) q[6];
ry(-1.5714701997754348) q[7];
rz(0.002100721526640647) q[7];
ry(-1.5708055536702743) q[8];
rz(-1.2266852532368153) q[8];
ry(1.5708246806756918) q[9];
rz(2.529969966333508) q[9];
ry(1.5706685422548274) q[10];
rz(-0.015060869187649423) q[10];
ry(-1.5708083153414365) q[11];
rz(0.06725467088165525) q[11];
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
ry(-2.3601816362473653) q[0];
rz(-1.2508532248918363) q[0];
ry(-1.5525276675698676) q[1];
rz(-0.33094061174697037) q[1];
ry(-3.14143319372403) q[2];
rz(2.457617940552364) q[2];
ry(3.10517679821372) q[3];
rz(-2.3474324994352123) q[3];
ry(1.585491321650999) q[4];
rz(1.359865627531792) q[4];
ry(-1.5688541898952633) q[5];
rz(1.138160060457368) q[5];
ry(-2.9522611648729504) q[6];
rz(2.169173608972616) q[6];
ry(-1.7428543156717602) q[7];
rz(1.5249331929248597) q[7];
ry(-0.0021880921182884094) q[8];
rz(-0.5362708723406808) q[8];
ry(-0.0001044318476521866) q[9];
rz(2.18237937594564) q[9];
ry(2.9576698038805844) q[10];
rz(1.6243200862681855) q[10];
ry(-3.1407400403331445) q[11];
rz(-0.0006263241917254803) q[11];
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
ry(2.3371073796538666) q[0];
rz(1.8058885821871797) q[0];
ry(-1.480251940537456) q[1];
rz(0.1868809478646991) q[1];
ry(3.141544556561565) q[2];
rz(-2.9248268759527623) q[2];
ry(2.0905138385373334) q[3];
rz(1.989555359988306) q[3];
ry(-1.8956565702055173e-05) q[4];
rz(-0.8184332182996362) q[4];
ry(0.00021088530478419187) q[5];
rz(-1.7053885784752607) q[5];
ry(0.001442074048958315) q[6];
rz(-0.9186732483398741) q[6];
ry(-0.011351093497450115) q[7];
rz(2.3073201864152733) q[7];
ry(-3.141282957064504) q[8];
rz(1.6631674948361548) q[8];
ry(2.739630008026347) q[9];
rz(3.1398761303444496) q[9];
ry(-0.0004994445577395886) q[10];
rz(1.212920826578404) q[10];
ry(-3.1415847057296906) q[11];
rz(3.0737173028173905) q[11];
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
ry(-2.6227226826080057) q[0];
rz(2.5578995035549528) q[0];
ry(1.576430159048039) q[1];
rz(-2.767160750602231) q[1];
ry(-3.1415638324756636) q[2];
rz(-1.1793833542595633) q[2];
ry(3.1401523932274893) q[3];
rz(-1.2071304811706136) q[3];
ry(-9.996469090900462e-05) q[4];
rz(2.418525635073996) q[4];
ry(3.140069960286351) q[5];
rz(0.3239312972048235) q[5];
ry(3.128885107585465) q[6];
rz(0.3172584917154521) q[6];
ry(3.1409898873522386) q[7];
rz(2.242968783117484) q[7];
ry(-1.0102567890696514) q[8];
rz(1.4166789197167655) q[8];
ry(-1.5683274807636658) q[9];
rz(-2.5408680002040316) q[9];
ry(-1.0435981610881757) q[10];
rz(1.7193828518858203) q[10];
ry(1.409272392263305) q[11];
rz(-0.2585542324573824) q[11];
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
ry(-6.074909499886872e-07) q[0];
rz(0.5896211790477706) q[0];
ry(4.821418805525207e-05) q[1];
rz(-1.6205967377805894) q[1];
ry(-3.14152869735542) q[2];
rz(-1.746435527462154) q[2];
ry(3.1405183299205106) q[3];
rz(-0.0693312613293875) q[3];
ry(3.141543697674215) q[4];
rz(-1.612901949733109) q[4];
ry(-3.1415306616945062) q[5];
rz(0.8821252352028539) q[5];
ry(-7.604465487679434e-06) q[6];
rz(-2.9567331511913113) q[6];
ry(0.0010996113593462198) q[7];
rz(1.5884052291664066) q[7];
ry(1.5708491321522708) q[8];
rz(-0.7335515261235557) q[8];
ry(1.5632657543721997) q[9];
rz(-0.005038983639646055) q[9];
ry(-1.5708419686123316) q[10];
rz(1.3582935233677391) q[10];
ry(2.167471864211734e-06) q[11];
rz(-1.6476602127755857) q[11];
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
ry(-1.1462832472520885) q[0];
rz(1.386900741660921) q[0];
ry(-0.4822753827006574) q[1];
rz(-1.8900624725170625) q[1];
ry(-3.1414286691108635) q[2];
rz(0.09489844801730563) q[2];
ry(1.5616079336118416) q[3];
rz(0.3093652201570265) q[3];
ry(1.5502116825152017) q[4];
rz(-0.019013960723021306) q[4];
ry(1.5684976727099214) q[5];
rz(1.5654951250608888) q[5];
ry(0.6529299950571136) q[6];
rz(-1.7304860425348538) q[6];
ry(1.7392331058687076) q[7];
rz(-0.0021190541718860622) q[7];
ry(0.000884964194671182) q[8];
rz(2.2929278734056546) q[8];
ry(1.5739573129285351) q[9];
rz(-1.6095702811492298) q[9];
ry(-3.138916624190226) q[10];
rz(-0.20874495772239718) q[10];
ry(1.2123642625899787e-05) q[11];
rz(0.3354262662461174) q[11];
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
ry(3.140849031763602) q[0];
rz(-1.620633992176792) q[0];
ry(3.140352954508175) q[1];
rz(-1.9608921685668865) q[1];
ry(3.141533946883613) q[2];
rz(0.07711343283998405) q[2];
ry(1.5710029501216196) q[3];
rz(2.693725238028996) q[3];
ry(0.7943821852187867) q[4];
rz(1.5710571278101293) q[4];
ry(-1.5717248504794474) q[5];
rz(0.0007901218672735195) q[5];
ry(-1.5917350210297716) q[6];
rz(-1.5781567227249342) q[6];
ry(-1.5708142752691368) q[7];
rz(-2.6634325532169645) q[7];
ry(2.8770565877499847) q[8];
rz(-0.011802138941697573) q[8];
ry(-3.0474450632720704) q[9];
rz(3.1027108039440225) q[9];
ry(3.1304823133911825) q[10];
rz(-3.137859820332758) q[10];
ry(-1.0700461920061697) q[11];
rz(-1.5708001440452306) q[11];
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
ry(-3.1413597444726578) q[0];
rz(-2.077968018308355) q[0];
ry(-3.1413677269243236) q[1];
rz(0.4432826753066587) q[1];
ry(6.848314312080106e-06) q[2];
rz(-1.6508605451979093) q[2];
ry(-3.141586810713245) q[3];
rz(2.392602845810168) q[3];
ry(1.5770299252977356) q[4];
rz(1.5714614927245147) q[4];
ry(1.5720941799922707) q[5];
rz(-1.56798162220787) q[5];
ry(0.8108726393958916) q[6];
rz(-2.8677382756112864) q[6];
ry(1.5717696449185325) q[7];
rz(-0.0020404989768427386) q[7];
ry(-1.561835646701115) q[8];
rz(2.9802793374714143) q[8];
ry(1.5786243202834696) q[9];
rz(-1.5716630923193264) q[9];
ry(-1.5785288460687656) q[10];
rz(1.57097868291714) q[10];
ry(2.392117848731104) q[11];
rz(1.6870829942172716) q[11];
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
ry(3.141304766978974) q[0];
rz(2.3261304769610054) q[0];
ry(-0.0002234351144405622) q[1];
rz(-2.503110448885255) q[1];
ry(3.141435555187248) q[2];
rz(2.220833480115281) q[2];
ry(3.14001270152613) q[3];
rz(1.281729180413297) q[3];
ry(1.5690777265180182) q[4];
rz(-3.1215169317689133) q[4];
ry(-1.5697487928949947) q[5];
rz(1.5313055570625205) q[5];
ry(3.130187652487157) q[6];
rz(0.876883232589681) q[6];
ry(1.5701565506865072) q[7];
rz(-1.6272436629740463) q[7];
ry(-0.00035294333355443596) q[8];
rz(-0.507923562299526) q[8];
ry(-1.5725863551672516) q[9];
rz(-1.5714118397458403) q[9];
ry(1.568557672293507) q[10];
rz(-1.5710305468400882) q[10];
ry(3.1408170210042914) q[11];
rz(1.0083824835908768) q[11];
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
ry(3.0459561328355327) q[0];
rz(0.3834562769783797) q[0];
ry(-2.873888230064885) q[1];
rz(2.6467031122835074) q[1];
ry(-1.5701602255476634) q[2];
rz(0.00015944859932215394) q[2];
ry(-2.874571141682585) q[3];
rz(-0.5273548288848598) q[3];
ry(-0.09810944104989615) q[4];
rz(-2.3966447232091372) q[4];
ry(-0.03377805473541339) q[5];
rz(0.0968838899432272) q[5];
ry(3.141210351989503) q[6];
rz(1.3468917386387769) q[6];
ry(2.230757990862031e-05) q[7];
rz(-0.9040010230280648) q[7];
ry(0.0007122008549523286) q[8];
rz(-2.454983724041085) q[8];
ry(1.5713285836521482) q[9];
rz(2.8205657780727758) q[9];
ry(1.571653507005974) q[10];
rz(2.903011917582974) q[10];
ry(-3.965404371086834e-06) q[11];
rz(2.2494995959211987) q[11];
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
ry(1.7977591635096767e-05) q[0];
rz(1.1829135482389581) q[0];
ry(7.5919391384735e-05) q[1];
rz(-0.24663090695177647) q[1];
ry(1.5713512835023107) q[2];
rz(3.140084743612632) q[2];
ry(-0.0001386239935395794) q[3];
rz(-2.607554438860463) q[3];
ry(3.0245814702567486e-05) q[4];
rz(0.8357705630290582) q[4];
ry(0.00014581247347233983) q[5];
rz(-0.05828032403799188) q[5];
ry(-3.141563446204426) q[6];
rz(0.7438853844428167) q[6];
ry(3.1415918341319946) q[7];
rz(0.6100931329907837) q[7];
ry(3.1415632965837146) q[8];
rz(0.017362347790674745) q[8];
ry(3.141550117369276) q[9];
rz(-2.7391051344468815) q[9];
ry(-2.6900410308172692e-05) q[10];
rz(0.4389796930500976) q[10];
ry(1.5706557902036566) q[11];
rz(-1.5675689841141658) q[11];
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
ry(-0.00028622745400453515) q[0];
rz(1.389892635384096) q[0];
ry(0.0006777134704178103) q[1];
rz(0.6740615734580703) q[1];
ry(1.5728270143480878) q[2];
rz(-3.1405397654999727) q[2];
ry(-1.570336042781737) q[3];
rz(0.004289103101024492) q[3];
ry(-3.134014104547113) q[4];
rz(1.604764219614423) q[4];
ry(-0.7758382886557454) q[5];
rz(0.02667901001724804) q[5];
ry(-1.5705558465641771) q[6];
rz(1.576813031515881) q[6];
ry(1.5707477900583937) q[7];
rz(0.0005989510605634612) q[7];
ry(-1.5701236736216677) q[8];
rz(-1.5707756893246998) q[8];
ry(3.140474777443818) q[9];
rz(-0.8472735565692666) q[9];
ry(-0.0015015619020477544) q[10];
rz(1.3704009852419092) q[10];
ry(2.978813456697485) q[11];
rz(1.5740117711561363) q[11];