OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(4.09068377393742e-06) q[0];
rz(-3.1262897545118946) q[0];
ry(3.1382560605427257) q[1];
rz(2.772259522106785) q[1];
ry(0.007870028025915292) q[2];
rz(-1.5667737138121156) q[2];
ry(3.1318647763586065) q[3];
rz(-1.8764438591199228) q[3];
ry(0.04340911994953966) q[4];
rz(-1.5760188767855619) q[4];
ry(0.16758847642709984) q[5];
rz(-1.570534285183361) q[5];
ry(0.6619064539967108) q[6];
rz(1.570909632574413) q[6];
ry(-1.5708031981479627) q[7];
rz(-2.9828794306785413) q[7];
ry(0.5452157380556857) q[8];
rz(-1.570729305012701) q[8];
ry(0.05894034694988016) q[9];
rz(-1.571603899236293) q[9];
ry(-3.1313846019531897) q[10];
rz(-0.0006993173510991823) q[10];
ry(3.141421000879863) q[11];
rz(-0.14065496843534464) q[11];
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
ry(-3.1391128724333415) q[0];
rz(1.602870063205538) q[0];
ry(0.00013774542312630395) q[1];
rz(-2.760896141662428) q[1];
ry(0.008633430122927166) q[2];
rz(1.5675973733420407) q[2];
ry(-0.001555436934905785) q[3];
rz(-1.2657052470464167) q[3];
ry(0.007638294199208494) q[4];
rz(-1.5654982587243118) q[4];
ry(0.029344047092635606) q[5];
rz(1.5703448100780035) q[5];
ry(-0.08888357911370812) q[6];
rz(-1.5839410795859115) q[6];
ry(1.5707728997930763) q[7];
rz(3.139825277591705) q[7];
ry(-2.8993693551959434) q[8];
rz(1.5717886588085859) q[8];
ry(0.9019096647761139) q[9];
rz(-1.5706018929433565) q[9];
ry(-1.5708028023014065) q[10];
rz(-0.09402440063539602) q[10];
ry(2.677681558743283) q[11];
rz(-1.5705956930682947) q[11];
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
ry(-0.3356561008066139) q[0];
rz(-6.146643229953952e-07) q[0];
ry(2.2243120229767372) q[1];
rz(-4.401532852860157e-06) q[1];
ry(1.068337655401156) q[2];
rz(-3.1415892951831474) q[2];
ry(0.5882415468964775) q[3];
rz(-1.670270143439012e-05) q[3];
ry(-0.24726869841711357) q[4];
rz(-1.0428812417195843e-05) q[4];
ry(0.07656343566060109) q[5];
rz(-3.1414251857911353) q[5];
ry(-0.009595843560653069) q[6];
rz(-3.12852952159199) q[6];
ry(3.0911392548929073) q[7];
rz(-0.001805369858202788) q[7];
ry(3.1059449858885357) q[8];
rz(-3.1406254055855576) q[8];
ry(0.06439999576160815) q[9];
rz(-0.00011385906912320111) q[9];
ry(1.5707793821249512) q[10];
rz(-1.5708613881130136) q[10];
ry(3.1106202591648655) q[11];
rz(0.00013953403697186673) q[11];
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
ry(-0.30023515349632035) q[0];
rz(-2.792842616860507) q[0];
ry(-2.325907479026306) q[1];
rz(-2.7928473345619955) q[1];
ry(-2.6185705167050957) q[2];
rz(0.34874945110942873) q[2];
ry(0.767412767933272) q[3];
rz(-2.79284433758141) q[3];
ry(-2.18736407703286) q[4];
rz(0.3487513562287159) q[4];
ry(-2.209907463245268) q[5];
rz(-2.792834137897957) q[5];
ry(-2.1467315752646328) q[6];
rz(0.34873760980461926) q[6];
ry(1.4460670645624374) q[7];
rz(0.3487668809088609) q[7];
ry(-2.3183709202896385) q[8];
rz(0.3487636827647389) q[8];
ry(-2.343180633958952) q[9];
rz(-2.7928102942303368) q[9];
ry(-1.0256299182587543) q[10];
rz(0.3487800511340362) q[10];
ry(0.32791683954042394) q[11];
rz(-2.7928144179559298) q[11];