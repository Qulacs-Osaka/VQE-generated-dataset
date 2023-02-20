OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.5690865378532113) q[0];
rz(1.6214639382543101) q[0];
ry(-1.5699382068643253) q[1];
rz(0.7565047381523051) q[1];
ry(-0.2420627099613144) q[2];
rz(-2.3002722659105217) q[2];
ry(-0.06489261038058203) q[3];
rz(-0.2774863977812194) q[3];
ry(1.5675794596010295) q[4];
rz(1.4978902552737374) q[4];
ry(1.5985845659292774) q[5];
rz(-0.6298452527422365) q[5];
ry(0.3253187007802197) q[6];
rz(-2.988548971780929) q[6];
ry(-3.1215605319407915) q[7];
rz(-1.1640825577073246) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.5876045254781919) q[0];
rz(1.4898230869820592) q[0];
ry(-3.0692688243307122) q[1];
rz(-2.3676909213366772) q[1];
ry(-6.873068372729893e-05) q[2];
rz(-1.1965072654313555) q[2];
ry(1.5696063326627219) q[3];
rz(-2.9036014947187394) q[3];
ry(-1.5631360760246447) q[4];
rz(1.8460154761321177) q[4];
ry(0.30330187897061944) q[5];
rz(0.35028940468440706) q[5];
ry(1.562361909860413) q[6];
rz(1.7336864172869433) q[6];
ry(0.007069064601763996) q[7];
rz(1.1768362761901032) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.8545702427855169) q[0];
rz(-1.1845561883716522) q[0];
ry(-2.2957395076028977) q[1];
rz(-3.112681849656361) q[1];
ry(0.23447212389726543) q[2];
rz(-0.03651169982765287) q[2];
ry(0.005469015930581565) q[3];
rz(2.9071087302034564) q[3];
ry(-1.7739218406344337) q[4];
rz(0.14204215774776507) q[4];
ry(-2.3264201409477794) q[5];
rz(2.3794370720390976) q[5];
ry(-1.3391301741882131) q[6];
rz(2.875401861129947) q[6];
ry(0.00021478111154316082) q[7];
rz(-1.588297893659751) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.007437968653884575) q[0];
rz(1.4637057494964578) q[0];
ry(-2.720876041685384) q[1];
rz(-3.118841668539068) q[1];
ry(3.1403762203731524) q[2];
rz(-1.9034517944257408) q[2];
ry(-1.5693039726263098) q[3];
rz(-0.040688509770459014) q[3];
ry(-1.5784276333877967) q[4];
rz(1.60494322643813) q[4];
ry(3.0821630761741674) q[5];
rz(2.570413009354401) q[5];
ry(1.5713219922138286) q[6];
rz(-0.02296997301428209) q[6];
ry(-0.003202943576709139) q[7];
rz(-1.5785865148236624) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.4805519289862565) q[0];
rz(-1.4182398559092366) q[0];
ry(1.5952673040912977) q[1];
rz(-1.4473429776655795) q[1];
ry(-0.026962421334150995) q[2];
rz(-1.2772709108343068) q[2];
ry(3.130213982471407) q[3];
rz(-5.877373415152931e-05) q[3];
ry(-1.5589222750359573) q[4];
rz(0.16132339190713468) q[4];
ry(-1.559770113755639) q[5];
rz(-3.1220391694821674) q[5];
ry(1.5994161769864874) q[6];
rz(1.5696122374512744) q[6];
ry(2.915001138600188) q[7];
rz(-1.5837406853015166) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.5783321126278465) q[0];
rz(-0.004889979674904637) q[0];
ry(3.1413887343493) q[1];
rz(-3.010772867721128) q[1];
ry(1.581011960110904) q[2];
rz(0.02160730026143773) q[2];
ry(1.585669225101185) q[3];
rz(0.19655417765834837) q[3];
ry(1.5658430448495013) q[4];
rz(-0.02255245483954086) q[4];
ry(1.56522915885433) q[5];
rz(-0.018561264204945083) q[5];
ry(1.5837464672164625) q[6];
rz(1.5477785519454959) q[6];
ry(1.5720270835031689) q[7];
rz(3.1329332394517966) q[7];