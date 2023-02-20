OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(3.140115236857154) q[0];
rz(2.4499812536964503) q[0];
ry(0.6119650228801086) q[1];
rz(-0.014990148463147612) q[1];
ry(0.7298367420841103) q[2];
rz(-0.4266679060045782) q[2];
ry(-0.20130280303477657) q[3];
rz(0.38062180945457236) q[3];
ry(-0.0001073180245461103) q[4];
rz(2.057466925069914) q[4];
ry(-1.5706754762298556) q[5];
rz(-3.031774355129839) q[5];
ry(0.34044175624260115) q[6];
rz(0.015287068811583815) q[6];
ry(-1.5705533497094457) q[7];
rz(-1.5643492688508267) q[7];
ry(-1.5727497399819161) q[8];
rz(-0.00033013927419123235) q[8];
ry(1.5899859775653864) q[9];
rz(3.1406353490688965) q[9];
ry(1.5552913238690398) q[10];
rz(-0.7774278829933721) q[10];
ry(-1.5662217512825882) q[11];
rz(0.8979678924149048) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(3.1411489047558847) q[0];
rz(-0.8894763000137925) q[0];
ry(-1.553056833424978) q[1];
rz(0.6708428334533632) q[1];
ry(-0.6250979367454557) q[2];
rz(0.5336477396164943) q[2];
ry(-0.007682895296895691) q[3];
rz(-2.9872272381272724) q[3];
ry(-2.4738735628915265e-06) q[4];
rz(-2.6595539272003608) q[4];
ry(-1.5650400763105783) q[5];
rz(-2.5436422991664074) q[5];
ry(2.9811849957650796) q[6];
rz(0.013309115363433932) q[6];
ry(-1.5708747902891493) q[7];
rz(-1.5876805577048012) q[7];
ry(-1.5696582426653292) q[8];
rz(3.1378319543867788) q[8];
ry(-1.5713896818714428) q[9];
rz(0.32242227750200564) q[9];
ry(-0.001181688400945659) q[10];
rz(2.3483546661868315) q[10];
ry(0.006003155642145152) q[11];
rz(2.74013039685304) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-2.2146145842742544) q[0];
rz(2.5335987567944778) q[0];
ry(0.7034550993341069) q[1];
rz(-1.5707772201992158) q[1];
ry(-2.3689989430884064) q[2];
rz(-2.510688718707614) q[2];
ry(2.307070935335466) q[3];
rz(0.1576318640397307) q[3];
ry(9.577503903636292e-05) q[4];
rz(-1.5207765930650872) q[4];
ry(-3.140210518645065) q[5];
rz(2.137874820370523) q[5];
ry(-1.570615841941118) q[6];
rz(-1.3676456445934528) q[6];
ry(0.22302482971342286) q[7];
rz(-3.124000518171919) q[7];
ry(-2.405510595231702) q[8];
rz(0.009126147103292402) q[8];
ry(-1.5732618984672806) q[9];
rz(0.44488250764956927) q[9];
ry(-0.797735228436136) q[10];
rz(1.8702219558845932) q[10];
ry(-1.5058922313268568) q[11];
rz(-0.12022256224985589) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.0008436241214342388) q[0];
rz(0.6071567626796687) q[0];
ry(-2.1183910901001468e-05) q[1];
rz(-1.2445098710209779) q[1];
ry(-3.140284184193509) q[2];
rz(-2.9576432659538923) q[2];
ry(1.486302250590291) q[3];
rz(2.0405984611246764) q[3];
ry(-2.9981167645475066) q[4];
rz(1.8648241520947026) q[4];
ry(2.9443155170971202) q[5];
rz(2.8794197925569556) q[5];
ry(-1.5983804697256332) q[6];
rz(0.19310504356608116) q[6];
ry(-2.3083936209486065) q[7];
rz(3.1223886858401855) q[7];
ry(1.5136171469527013) q[8];
rz(-0.0011195075122879278) q[8];
ry(-0.0020681184310546996) q[9];
rz(-0.4502953105169824) q[9];
ry(-3.084652511983355) q[10];
rz(-2.1181574650301274) q[10];
ry(1.566908030547948) q[11];
rz(-1.5747804177231373) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.927735697925562) q[0];
rz(1.4962634658179441) q[0];
ry(-3.1392113049336627) q[1];
rz(2.893034659362281) q[1];
ry(-1.2658712305750113) q[2];
rz(-3.1298822706596465) q[2];
ry(-1.180932847771806) q[3];
rz(2.7899006881128785) q[3];
ry(-3.141478045590727) q[4];
rz(1.8683386005358713) q[4];
ry(5.9613439522233225e-05) q[5];
rz(0.9682653723120067) q[5];
ry(-3.137040727539361) q[6];
rz(1.9087808366264332) q[6];
ry(1.4036251610340855) q[7];
rz(3.1355514884105307) q[7];
ry(3.1380358893298173) q[8];
rz(0.11334742242471744) q[8];
ry(0.21199549302209422) q[9];
rz(1.164562169875567) q[9];
ry(-3.1370989851003497) q[10];
rz(2.2971559537576725) q[10];
ry(1.5727316263428248) q[11];
rz(1.279716972101947) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.370252086762335e-05) q[0];
rz(0.9404936427339213) q[0];
ry(0.005793006742460496) q[1];
rz(1.1092830620632395) q[1];
ry(-1.623691410327667) q[2];
rz(-3.1394982474198168) q[2];
ry(-0.7164849936232986) q[3];
rz(-1.6523663305690803) q[3];
ry(-1.5060749189676486) q[4];
rz(-2.651877780258007) q[4];
ry(-3.140610483233694) q[5];
rz(-2.65266612866241) q[5];
ry(2.2443774253986293) q[6];
rz(2.7649288287340394) q[6];
ry(2.8899361028243606) q[7];
rz(3.1346159532096816) q[7];
ry(1.6273056519002906) q[8];
rz(-1.573592418369759) q[8];
ry(-0.0036983079182094157) q[9];
rz(-1.1583467120537447) q[9];
ry(1.570685387109066) q[10];
rz(3.141574803418544) q[10];
ry(-2.6386429748348394) q[11];
rz(-1.8262983360483283) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-2.3687661094600907) q[0];
rz(1.570429324279709) q[0];
ry(1.5697372309073234) q[1];
rz(1.3437603972291694) q[1];
ry(-1.4200552689045054) q[2];
rz(-0.015267211229780035) q[2];
ry(-2.4098820345789442) q[3];
rz(-2.0757182717818594) q[3];
ry(3.1415842183261735) q[4];
rz(-2.649911827945464) q[4];
ry(-3.140640727495264) q[5];
rz(1.324811749553282) q[5];
ry(3.1412918281977125) q[6];
rz(-1.8703340171806413) q[6];
ry(-1.3070774933753926) q[7];
rz(1.7561561627085291) q[7];
ry(-1.5717531517053427) q[8];
rz(0.0014663090261707867) q[8];
ry(-1.561997662685141) q[9];
rz(1.5711034171527458) q[9];
ry(1.572263006310596) q[10];
rz(-3.035739709975933) q[10];
ry(1.5840491544189437) q[11];
rz(0.005370299498514086) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.570253562433422) q[0];
rz(-2.4739425047000934) q[0];
ry(-0.00015803293323646272) q[1];
rz(0.8522827262535069) q[1];
ry(1.5729707472497008) q[2];
rz(-0.8882865864309933) q[2];
ry(-6.302158563630442e-05) q[3];
rz(1.8868385377725314) q[3];
ry(2.7449676615471064) q[4];
rz(2.329503102925704) q[4];
ry(-1.5701104020452423) q[5];
rz(0.6307193142668366) q[5];
ry(-1.5712105132611918) q[6];
rz(2.231611161263139) q[6];
ry(3.139843401343495) q[7];
rz(2.375703887665368) q[7];
ry(1.5697919857313687) q[8];
rz(-2.483387957872371) q[8];
ry(1.5682375885038264) q[9];
rz(-0.9660287227577297) q[9];
ry(1.570626746432838) q[10];
rz(-0.91568818447962) q[10];
ry(1.5704830848296787) q[11];
rz(2.1767543171936876) q[11];