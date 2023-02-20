OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.570794380712071) q[0];
rz(3.141592231532551) q[0];
ry(-3.1415923497416784) q[1];
rz(0.5723104141127201) q[1];
ry(1.7812470202402926) q[2];
rz(3.077067494384638) q[2];
ry(3.1415917107980285) q[3];
rz(-3.0840418482253016) q[3];
ry(-1.5707913125265967) q[4];
rz(3.1415910595474164) q[4];
ry(-1.5707968160837193) q[5];
rz(3.141590677541662) q[5];
ry(-1.570795617385164) q[6];
rz(-0.6872763090777283) q[6];
ry(-2.645648811906495) q[7];
rz(-1.5707965293126942) q[7];
ry(1.5707969845087768) q[8];
rz(-0.7907818312164947) q[8];
ry(1.5809077229533841e-06) q[9];
rz(2.578257379941467) q[9];
ry(-1.5707962720004183) q[10];
rz(-1.570796211871162) q[10];
ry(-1.7039271846478155) q[11];
rz(-3.141592408966549) q[11];
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
ry(1.005700849788079) q[0];
rz(0.7402069293941329) q[0];
ry(-3.1415916038328318) q[1];
rz(1.9488061925399789) q[1];
ry(-3.141590576110481) q[2];
rz(-2.7840540345418945) q[2];
ry(-1.0907192516011641e-06) q[3];
rz(2.5660846675687954) q[3];
ry(-0.31262549966451436) q[4];
rz(1.5707978355968208) q[4];
ry(-2.094256722825177) q[5];
rz(1.5707947304920464) q[5];
ry(1.7781914003478505e-06) q[6];
rz(0.2562429700345729) q[6];
ry(-1.5707976553253908) q[7];
rz(-3.141583700708585) q[7];
ry(-3.141592573207875) q[8];
rz(-0.7907820137455992) q[8];
ry(1.570796149670207) q[9];
rz(-1.6462098234131561) q[9];
ry(1.570796032008432) q[10];
rz(-2.7747795635324572) q[10];
ry(-2.156513095495206) q[11];
rz(-5.663499971220176e-07) q[11];
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
ry(-3.1415853765231376) q[0];
rz(2.3110030457964315) q[0];
ry(1.3881002391968877e-07) q[1];
rz(-0.5584990555322401) q[1];
ry(-7.090511706664415e-07) q[2];
rz(2.7195292703329597) q[2];
ry(-0.09515689903164588) q[3];
rz(-1.5708001757557442) q[3];
ry(-1.4662827533606846) q[4];
rz(-1.5314084529066272e-07) q[4];
ry(-1.256044607348127) q[5];
rz(1.5707917311865771) q[5];
ry(-3.1415909609446486) q[6];
rz(-2.001829488108754) q[6];
ry(1.5707966595745468) q[7];
rz(-0.9303820902519142) q[7];
ry(-1.5707959505405023) q[8];
rz(-0.5875117595023501) q[8];
ry(-3.141592635577916) q[9];
rz(-0.1063016321633691) q[9];
ry(-1.570795207135606) q[10];
rz(3.141592234772572) q[10];
ry(-1.5707956073043423) q[11];
rz(2.817513799935006) q[11];
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
ry(-1.570795874285833) q[0];
rz(2.9505728910597564) q[0];
ry(-1.57079752908665) q[1];
rz(9.864155242889975e-08) q[1];
ry(1.5707890420016046) q[2];
rz(1.4104137857628984e-07) q[2];
ry(-1.570796733785935) q[3];
rz(-6.105062237433638e-07) q[3];
ry(1.5707982034510841) q[4];
rz(-1.9081240865109201) q[4];
ry(-1.539017644336432) q[5];
rz(3.141591762531427) q[5];
ry(1.5707968505581924) q[6];
rz(2.7996893875644076) q[6];
ry(-1.565917288859741e-06) q[7];
rz(-2.9389462277655953) q[7];
ry(1.2532326030267882e-06) q[8];
rz(1.252690937662254) q[8];
ry(-3.1415906904078867) q[9];
rz(-2.20119275533491) q[9];
ry(1.5707961346951937) q[10];
rz(-2.6101669677923174) q[10];
ry(2.418941758008941e-07) q[11];
rz(-2.3438676971032844) q[11];
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
ry(1.5707961028601254) q[0];
rz(3.097221015705261) q[0];
ry(-1.570802326829433) q[1];
rz(-1.4678313167852048) q[1];
ry(1.5707960703922192) q[2];
rz(1.8326186367456703) q[2];
ry(1.5707900465833557) q[3];
rz(0.477602551611632) q[3];
ry(1.4476930587647985e-06) q[4];
rz(1.8791839854081571) q[4];
ry(1.5707953709020446) q[5];
rz(0.9171209019144966) q[5];
ry(9.925805565647039e-05) q[6];
rz(-2.007943113239305) q[6];
ry(1.5793336505254274e-07) q[7];
rz(0.7277345530834712) q[7];
ry(-2.2715147551972499e-07) q[8];
rz(0.18816252088914265) q[8];
ry(3.141591844314289) q[9];
rz(0.9364482372024944) q[9];
ry(-3.1415911751039483) q[10];
rz(-2.6101867259557463) q[10];
ry(-3.1415923119196503) q[11];
rz(1.5909345311586034) q[11];
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
ry(-3.1415902150490416) q[0];
rz(-1.882119555495196) q[0];
ry(-2.984036360231812e-07) q[1];
rz(-0.3792801647217176) q[1];
ry(5.237294375959323e-07) q[2];
rz(-0.2618057491186665) q[2];
ry(3.1415923606492426) q[3];
rz(0.4357059461963804) q[3];
ry(-1.570796338069128) q[4];
rz(3.141592217790304) q[4];
ry(3.141591832302865) q[5];
rz(-2.2244730940501265) q[5];
ry(-3.141591744942516) q[6];
rz(0.6614381111937947) q[6];
ry(-1.0502986950734474) q[7];
rz(-0.8830311463464104) q[7];
ry(3.141592495612515) q[8];
rz(2.843399566910639) q[8];
ry(-2.4173292311953355e-07) q[9];
rz(-3.106752198794063) q[9];
ry(-3.1287563874094317) q[10];
rz(-1.5708156915160985) q[10];
ry(4.3153331397149946e-07) q[11];
rz(2.024303223186173) q[11];
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
ry(-6.580623908271984e-07) q[0];
rz(-2.8746401546843696) q[0];
ry(-3.1415925839605374) q[1];
rz(-0.27631515180762634) q[1];
ry(3.1314182412174025) q[2];
rz(-3.1415774663800686) q[2];
ry(-3.141592375071155) q[3];
rz(-1.59938502156839) q[3];
ry(-1.5707970681042065) q[4];
rz(-1.5707939140623566) q[4];
ry(1.622730861032804) q[5];
rz(-3.1415914268328793) q[5];
ry(-1.570796832671815) q[6];
rz(-0.24788076256098648) q[6];
ry(1.4878774798887662e-06) q[7];
rz(-0.6877655602172368) q[7];
ry(1.5707948838445105) q[8];
rz(2.9873677974349713) q[8];
ry(-1.5707963900577835) q[9];
rz(-1.570796025814136) q[9];
ry(1.57079627867762) q[10];
rz(-1.8415130636492132e-07) q[10];
ry(-1.5707963589635805) q[11];
rz(-0.058618908122479496) q[11];
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
ry(1.5707967916718033) q[0];
rz(-2.0066414155121954) q[0];
ry(1.5709207493521928) q[1];
rz(0.5046466552397142) q[1];
ry(-1.5707968756934427) q[2];
rz(-0.017761779513755016) q[2];
ry(0.0001247929035201878) q[3];
rz(-1.7433726303354042) q[3];
ry(1.5707988749607764) q[4];
rz(3.015465714848205) q[4];
ry(1.570795633331743) q[5];
rz(-1.5707971368923719) q[5];
ry(-1.0714109919844884e-06) q[6];
rz(-1.3229172573471617) q[6];
ry(1.5708733683320573) q[7];
rz(-1.5707964518969268) q[7];
ry(3.14103297836808) q[8];
rz(-0.15422487685697472) q[8];
ry(-1.5707947122668633) q[9];
rz(-1.2642363453373551e-07) q[9];
ry(-1.5707961964035961) q[10];
rz(2.870035900535952) q[10];
ry(-3.139875096295284) q[11];
rz(-1.6323447517834917) q[11];
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
ry(3.14158592779399) q[0];
rz(2.460492969609553) q[0];
ry(3.141588879449207) q[1];
rz(0.6240919847524546) q[1];
ry(3.141590950659613) q[2];
rz(-1.0584940379203116) q[2];
ry(3.141592045950836) q[3];
rz(2.3430437048805595) q[3];
ry(6.997311707834797e-08) q[4];
rz(0.07896216311787096) q[4];
ry(-1.5707964855180396) q[5];
rz(-1.8834295720658105) q[5];
ry(-1.5707949246692858) q[6];
rz(-0.45090760187314677) q[6];
ry(-1.5707981356936296) q[7];
rz(-1.260352982239505) q[7];
ry(1.57079698310159) q[8];
rz(7.924617129053047e-06) q[8];
ry(1.5707942708290261) q[9];
rz(6.473268889806151e-07) q[9];
ry(-3.141591793326362) q[10];
rz(1.299243290182745) q[10];
ry(-3.029965487399977e-06) q[11];
rz(-2.6232205857218536) q[11];
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
ry(-3.141592559456295) q[0];
rz(-0.24525481524170292) q[0];
ry(1.3419727206605796e-07) q[1];
rz(-0.11944518771024713) q[1];
ry(3.141592182298267) q[2];
rz(2.1008617391804663) q[2];
ry(-5.36896175518109e-07) q[3];
rz(2.2100792012232144) q[3];
ry(-3.0145118017723305e-07) q[4];
rz(1.6179600604433324) q[4];
ry(-3.141592368634676) q[5];
rz(-0.31263265360391) q[5];
ry(-7.187550155699911e-07) q[6];
rz(-2.4644652715544377) q[6];
ry(-3.141592383206558) q[7];
rz(-2.954849224827484) q[7];
ry(1.5707976356929292) q[8];
rz(-1.5707964354739916) q[8];
ry(-1.5707962137607252) q[9];
rz(-1.5707968696390164) q[9];
ry(-1.5707961627124265) q[10];
rz(-1.5707960419487188) q[10];
ry(3.1415924130501383) q[11];
rz(-1.1925432583548181) q[11];
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
ry(-1.5708032418038762) q[0];
rz(-2.9138063907138463) q[0];
ry(-1.570796120317092) q[1];
rz(0.6579350967554016) q[1];
ry(-1.570795657816359) q[2];
rz(1.798582315217735) q[2];
ry(-1.570795139308914) q[3];
rz(0.6579382628099103) q[3];
ry(-1.5707944103486993) q[4];
rz(0.22778284708209995) q[4];
ry(1.5707968538287826) q[5];
rz(0.6579255855111584) q[5];
ry(-5.617789060607947e-07) q[6];
rz(0.0015492070873692628) q[6];
ry(4.7894298818462744e-05) q[7];
rz(0.7816280458912914) q[7];
ry(-1.5707965988125494) q[8];
rz(-2.9139491101774224) q[8];
ry(1.5707960121187912) q[9];
rz(-0.9128599255224125) q[9];
ry(1.5707960680863287) q[10];
rz(0.2276410438997205) q[10];
ry(3.141590468830143) q[11];
rz(2.0915437378552255) q[11];