OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.3212229990044) q[0];
rz(-1.1896756306391296) q[0];
ry(-0.36866856562049755) q[1];
rz(-2.5536446524248686) q[1];
ry(-3.098370493962252) q[2];
rz(2.2473884295285096) q[2];
ry(-0.38200739462483113) q[3];
rz(1.5500802923955463) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.7940463166707586) q[0];
rz(-0.5673665946846153) q[0];
ry(2.777072313740502) q[1];
rz(1.5395147696292037) q[1];
ry(-2.936642018058178) q[2];
rz(-1.0245078739201174) q[2];
ry(1.8881076662872038) q[3];
rz(-1.788698678563641) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.7941316960352562) q[0];
rz(0.5913401734322878) q[0];
ry(1.1978738296762597) q[1];
rz(0.6833834403412349) q[1];
ry(-0.4671577287300994) q[2];
rz(0.38190762004782997) q[2];
ry(0.9763758968242077) q[3];
rz(-1.7527955735246157) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-3.04646508360819) q[0];
rz(-1.8030273903252858) q[0];
ry(-2.0578713808361044) q[1];
rz(-1.5620993404390318) q[1];
ry(-2.6207515709118567) q[2];
rz(2.871728870403312) q[2];
ry(-0.4159268916532195) q[3];
rz(0.9137313765726427) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.4191102914305881) q[0];
rz(-2.035751426615731) q[0];
ry(-1.5605563413671646) q[1];
rz(1.1427833316085927) q[1];
ry(-2.360253520589555) q[2];
rz(2.677944667836761) q[2];
ry(-1.475159894920398) q[3];
rz(-2.4709370720601296) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.5522724167912836) q[0];
rz(-2.2963632383816255) q[0];
ry(1.4774415912092032) q[1];
rz(-2.2803972279931477) q[1];
ry(1.8592629928529631) q[2];
rz(2.7882768815752956) q[2];
ry(1.9193033147123904) q[3];
rz(0.7722563370795277) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.8662524099249609) q[0];
rz(1.481021699107104) q[0];
ry(-2.767647856268153) q[1];
rz(-1.4193514406408512) q[1];
ry(-1.2130513057378518) q[2];
rz(-0.15004071380590567) q[2];
ry(0.6418352115326357) q[3];
rz(1.0399168771468181) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.5342900057114252) q[0];
rz(-1.614504667880853) q[0];
ry(2.0744422103850626) q[1];
rz(1.6751977541104104) q[1];
ry(-1.7929649869796964) q[2];
rz(-0.9029628853942147) q[2];
ry(-0.6389177813274536) q[3];
rz(-1.1459204190094026) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.4004144794513973) q[0];
rz(-2.93445990793322) q[0];
ry(-0.028368271236308473) q[1];
rz(-0.19270796734839912) q[1];
ry(-2.8880349691798295) q[2];
rz(1.922134989417331) q[2];
ry(2.916430275320625) q[3];
rz(0.5084302394141577) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.1848266620166911) q[0];
rz(-1.810715270461942) q[0];
ry(0.5951072593684461) q[1];
rz(-0.35647389090609677) q[1];
ry(3.0858075080464094) q[2];
rz(0.2771777222624001) q[2];
ry(-3.0952114108344886) q[3];
rz(-0.8204685591132224) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.6722294292060476) q[0];
rz(-3.0422995379552797) q[0];
ry(0.5745098625455896) q[1];
rz(-0.7373118767650457) q[1];
ry(-2.0425103572051215) q[2];
rz(0.5751491300114192) q[2];
ry(1.0509769460794107) q[3];
rz(2.208522557737753) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.991561288499673) q[0];
rz(0.120815330816937) q[0];
ry(-0.07085132112943157) q[1];
rz(0.8342575808490674) q[1];
ry(-1.4980269857509079) q[2];
rz(-2.036675394005812) q[2];
ry(-2.3263959746641127) q[3];
rz(-0.41358400314006266) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.6630966865871635) q[0];
rz(2.483137793711767) q[0];
ry(2.128276271393134) q[1];
rz(2.269984631076202) q[1];
ry(2.766057406772471) q[2];
rz(1.0625822662140485) q[2];
ry(2.6127653998976252) q[3];
rz(1.3820345730238361) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.2727750986253032) q[0];
rz(2.9790938716328292) q[0];
ry(-2.7052912495248216) q[1];
rz(3.0228510350477635) q[1];
ry(2.331250226281125) q[2];
rz(-1.5168110009304734) q[2];
ry(-2.3576462489218604) q[3];
rz(0.0543642745395676) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.4464105708351365) q[0];
rz(-1.8331787006235383) q[0];
ry(0.05519794639997195) q[1];
rz(-2.546555005812778) q[1];
ry(-0.5115896266371734) q[2];
rz(1.3096061244978137) q[2];
ry(-2.3942964103971662) q[3];
rz(-0.3624929611727709) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.2204548898610341) q[0];
rz(-0.648320218667114) q[0];
ry(-1.1039832121774609) q[1];
rz(-1.9932276300995218) q[1];
ry(-0.04969247408626032) q[2];
rz(-0.1801675017420077) q[2];
ry(2.936844810964576) q[3];
rz(-2.1054687031249277) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.525339619691765) q[0];
rz(2.537736129685803) q[0];
ry(1.8641748175886814) q[1];
rz(2.700098005236598) q[1];
ry(-1.1964540613351282) q[2];
rz(-0.0006468826069019329) q[2];
ry(1.2691429549845559) q[3];
rz(2.9851355842238885) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.6815441968099694) q[0];
rz(1.9010016522871185) q[0];
ry(-0.7831267801890834) q[1];
rz(1.6349339302383668) q[1];
ry(-0.2558870944334719) q[2];
rz(0.6072979286400471) q[2];
ry(0.9208388042557658) q[3];
rz(-0.6171436435766997) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.0580815523799783) q[0];
rz(-0.6479974432831774) q[0];
ry(-0.864347712404034) q[1];
rz(1.8650928284360662) q[1];
ry(-2.2160133711147645) q[2];
rz(2.193585034134956) q[2];
ry(-2.25892563904083) q[3];
rz(-2.8037313619361055) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.7232111272247232) q[0];
rz(-1.9723761273951155) q[0];
ry(-0.7632386166879828) q[1];
rz(0.18777041537289998) q[1];
ry(1.2038275359773907) q[2];
rz(-1.523494891526493) q[2];
ry(-0.11650128340962239) q[3];
rz(-0.8263878559470226) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.7305092625128815) q[0];
rz(0.2601419529270608) q[0];
ry(1.7963156648205723) q[1];
rz(2.7519755151931418) q[1];
ry(2.6544155849956113) q[2];
rz(3.1388995427547957) q[2];
ry(-2.9517107567989993) q[3];
rz(1.4051380634477337) q[3];