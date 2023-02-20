OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(9.522976096145481e-05) q[0];
rz(0.4864997123278812) q[0];
ry(0.0012410392781667667) q[1];
rz(-2.561693565338949) q[1];
ry(-1.5697713881901558) q[2];
rz(-0.0904125251480873) q[2];
ry(-0.2560805511673818) q[3];
rz(-0.6923431653099472) q[3];
ry(-1.3674340976974646) q[4];
rz(1.5856320535065107) q[4];
ry(-1.5710339237348077) q[5];
rz(-1.0879128384325487) q[5];
ry(4.6225973024386716e-05) q[6];
rz(1.8268661602679976) q[6];
ry(0.00029651408526909506) q[7];
rz(-3.1093319606738974) q[7];
ry(-1.0505123588408214) q[8];
rz(-0.021136171663353135) q[8];
ry(3.033026252675838) q[9];
rz(-1.3442943900779472) q[9];
ry(2.8484192415045264) q[10];
rz(-0.057834506318484004) q[10];
ry(-0.00026001166332629243) q[11];
rz(0.20614309789411855) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(3.1403146190929094) q[0];
rz(-0.8000622311259813) q[0];
ry(-0.0017954196427462743) q[1];
rz(-0.715046316604352) q[1];
ry(-1.615237841504747) q[2];
rz(1.5602886871266164) q[2];
ry(0.007442020877047818) q[3];
rz(-1.8042063546347151) q[3];
ry(-1.5710989384170078) q[4];
rz(-1.7013095514763528) q[4];
ry(-2.8449874168331055) q[5];
rz(-2.8642862182071367) q[5];
ry(-1.620302564802408) q[6];
rz(1.5727206341502171) q[6];
ry(1.5607091336561718) q[7];
rz(0.0010711714702624775) q[7];
ry(1.305669747828779) q[8];
rz(1.4900270377705214) q[8];
ry(1.5759733469601125) q[9];
rz(-2.889194558824104) q[9];
ry(3.010995821897222) q[10];
rz(-3.0756236404171142) q[10];
ry(0.0017653412933325906) q[11];
rz(1.0233301631852836) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.0575966897549427) q[0];
rz(-1.9160771130034102) q[0];
ry(2.875265253583102) q[1];
rz(2.3659429035617188) q[1];
ry(-1.820139907922771) q[2];
rz(1.30081755375533) q[2];
ry(0.0007210612015819962) q[3];
rz(-0.6486226299220313) q[3];
ry(3.138273569564292) q[4];
rz(-0.22262395184627692) q[4];
ry(-0.013428143527950631) q[5];
rz(0.09526073319393637) q[5];
ry(-1.5707483168411374) q[6];
rz(1.5660184467498155) q[6];
ry(-1.5641658757144778) q[7];
rz(1.5723402980700365) q[7];
ry(1.5687003858301392) q[8];
rz(3.1008595504172063) q[8];
ry(-1.312450332893289) q[9];
rz(3.080140193610474) q[9];
ry(1.5745418039662695) q[10];
rz(-0.20849520343252512) q[10];
ry(-0.2521446551992337) q[11];
rz(-1.5832391330233546) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(3.045805053300668) q[0];
rz(-3.1219224966509422) q[0];
ry(1.8417041410219444) q[1];
rz(-1.4747432650097918) q[1];
ry(-3.088704643943587) q[2];
rz(-0.931445668897199) q[2];
ry(-2.906633173015583) q[3];
rz(-3.0224393657823456) q[3];
ry(-1.5289432605824524) q[4];
rz(1.5709025134843968) q[4];
ry(-1.5710540879595674) q[5];
rz(1.5707972199798625) q[5];
ry(0.5400546442611098) q[6];
rz(1.5756086602766364) q[6];
ry(-0.11276677305607397) q[7];
rz(1.5690943472325225) q[7];
ry(1.5703851876840307) q[8];
rz(-0.12107795208440432) q[8];
ry(-1.570715174929318) q[9];
rz(3.1059650123409814) q[9];
ry(0.002177820302046385) q[10];
rz(1.9851798614190788) q[10];
ry(3.113161132702641) q[11];
rz(-2.78972387362569) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(3.1411684247988347) q[0];
rz(0.42468507943483064) q[0];
ry(0.09817560556256287) q[1];
rz(2.4745052570571144) q[1];
ry(0.0006656705100124322) q[2];
rz(-2.4640075556790215) q[2];
ry(-0.0013729947249458974) q[3];
rz(2.6908788049483463) q[3];
ry(1.5717664973967105) q[4];
rz(-3.104807262181708) q[4];
ry(1.570148181216758) q[5];
rz(0.11424108520455079) q[5];
ry(-1.4423565753173715) q[6];
rz(-0.39866836985058324) q[6];
ry(-1.5709255154095507) q[7];
rz(-2.253794243316591) q[7];
ry(-1.5843539720056476) q[8];
rz(-2.977285866233903) q[8];
ry(-1.5718647570860107) q[9];
rz(-2.077264324435336) q[9];
ry(3.047639228581736) q[10];
rz(-1.358780993548125) q[10];
ry(3.1381877911600076) q[11];
rz(-0.15573523937600342) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.4627249952654768) q[0];
rz(-0.04286314258411847) q[0];
ry(0.5856688990943936) q[1];
rz(2.149741637099037) q[1];
ry(1.5652679440706372) q[2];
rz(3.100262605284018) q[2];
ry(3.140018215818258) q[3];
rz(2.818139809010191) q[3];
ry(-1.5630293289904884) q[4];
rz(-3.139026649874293) q[4];
ry(1.5698438675201816) q[5];
rz(-3.096807726578152) q[5];
ry(0.00013544724629088734) q[6];
rz(1.939661813788173) q[6];
ry(0.0026932090378517515) q[7];
rz(0.6831795371391056) q[7];
ry(0.0002955493093059758) q[8];
rz(1.517991980731109) q[8];
ry(-0.0023506113374898168) q[9];
rz(-2.632243096262846) q[9];
ry(-1.8772169651819466) q[10];
rz(-2.789489327996246) q[10];
ry(-0.0004658804730673004) q[11];
rz(2.7382869793522855) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.872570145124136) q[0];
rz(-1.6308290811532047) q[0];
ry(-0.025497637068284867) q[1];
rz(-1.4134037906582815) q[1];
ry(0.020821878787067157) q[2];
rz(1.606861927033026) q[2];
ry(3.0830847863825173) q[3];
rz(1.5693546785549854) q[3];
ry(3.1211384045757384) q[4];
rz(-0.01953109557554544) q[4];
ry(-3.1307318605543384) q[5];
rz(-3.0959900682042) q[5];
ry(-1.5707329136036068) q[6];
rz(0.4134774491470544) q[6];
ry(1.553086347297454) q[7];
rz(1.5668045885429664) q[7];
ry(2.756884956976445) q[8];
rz(0.3660980669762397) q[8];
ry(-3.023715925486752) q[9];
rz(3.1128995359458753) q[9];
ry(-0.03538700332112055) q[10];
rz(2.7774157338609773) q[10];
ry(0.007292391421911154) q[11];
rz(2.5061901223327885) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.812135565526787) q[0];
rz(-1.6276829657960354) q[0];
ry(-2.797956789070943) q[1];
rz(1.7017461341453293) q[1];
ry(1.5515463541040777) q[2];
rz(3.1363297599003817) q[2];
ry(-1.5627979953975535) q[3];
rz(3.13903095131647) q[3];
ry(1.5764709562359316) q[4];
rz(-1.5738571880347112) q[4];
ry(-1.6040376989352625) q[5];
rz(3.125665538015324) q[5];
ry(3.139318958925131) q[6];
rz(-1.16237930726692) q[6];
ry(-1.5654712571181704) q[7];
rz(1.557588141823867) q[7];
ry(3.1402617072211054) q[8];
rz(1.941987553728849) q[8];
ry(1.5343389162071803) q[9];
rz(-3.141377037433445) q[9];
ry(1.9114030347500968) q[10];
rz(3.1370234416683607) q[10];
ry(1.568781477989589) q[11];
rz(-3.141247581559055) q[11];