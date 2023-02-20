OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(3.139664190297774) q[0];
rz(2.3031264617723752) q[0];
ry(-0.5261487239564318) q[1];
rz(-1.6928302880627364) q[1];
ry(-3.1412075584464323) q[2];
rz(-1.6669966836175503) q[2];
ry(0.10133295444080126) q[3];
rz(-1.4507789488849523) q[3];
ry(1.5708167388200875) q[4];
rz(3.1414609359272623) q[4];
ry(0.00021183162538884756) q[5];
rz(2.7538873988045918) q[5];
ry(2.4176662655802326) q[6];
rz(2.123615016734108) q[6];
ry(-1.5700284077300302) q[7];
rz(3.1410054506561234) q[7];
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
ry(0.0030604738730035663) q[0];
rz(-0.5790595964050239) q[0];
ry(-1.5099529977775452) q[1];
rz(0.5335721368642341) q[1];
ry(0.053290241893663826) q[2];
rz(-1.5481559775078093) q[2];
ry(-1.4012008361495258) q[3];
rz(1.6025101221552052) q[3];
ry(1.5712566275707731) q[4];
rz(2.384757884429447) q[4];
ry(-2.6872150220226256) q[5];
rz(-2.6942998085109844) q[5];
ry(-1.9651470563818503) q[6];
rz(-2.901692586647518) q[6];
ry(1.569956209134638) q[7];
rz(0.4110813011721759) q[7];
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
ry(1.5717417659788324) q[0];
rz(0.6922140451990761) q[0];
ry(-0.10595983028932798) q[1];
rz(-2.670071855223755) q[1];
ry(2.2930727761525724) q[2];
rz(1.6762945644682257) q[2];
ry(5.803099790036015e-06) q[3];
rz(-3.0520449748625738) q[3];
ry(0.00011921620817911815) q[4];
rz(-2.5992574583325925) q[4];
ry(0.08889454896742013) q[5];
rz(-1.1238490715526446) q[5];
ry(1.86621535585089) q[6];
rz(0.33579084839620743) q[6];
ry(1.5704977625341687) q[7];
rz(0.41816129401255187) q[7];
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
ry(-3.1415771331400855) q[0];
rz(0.6414276340828261) q[0];
ry(0.050901373258210265) q[1];
rz(1.6192309270380267) q[1];
ry(-1.5709160423360107) q[2];
rz(3.1415666038232075) q[2];
ry(0.11196539912802717) q[3];
rz(0.7159414105159687) q[3];
ry(-1.5711175668623607) q[4];
rz(1.5709458957857023) q[4];
ry(0.00019866205120482033) q[5];
rz(-2.971185215120863) q[5];
ry(1.5708906347480038) q[6];
rz(-0.8983853047701497) q[6];
ry(-1.9171058807168597) q[7];
rz(0.00030709088124147854) q[7];
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
ry(2.4206278711492693) q[0];
rz(2.7800710567968308) q[0];
ry(-1.5719144121015045) q[1];
rz(-0.8944066180437965) q[1];
ry(-1.57044172991389) q[2];
rz(2.525880791726515) q[2];
ry(1.570801764241585) q[3];
rz(-1.570580526528481) q[3];
ry(0.7046439093292182) q[4];
rz(3.141290638567931) q[4];
ry(-3.141576492410549) q[5];
rz(-1.5456784467714364) q[5];
ry(-3.14068770905938) q[6];
rz(-1.976223175443008) q[6];
ry(2.3679578681739293) q[7];
rz(-0.5589445133231523) q[7];
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
ry(3.139569645790092) q[0];
rz(2.6820891964701357) q[0];
ry(1.4598197230123762) q[1];
rz(1.2890116410079093) q[1];
ry(3.1415727870025303) q[2];
rz(0.18560606785496048) q[2];
ry(-0.0699731356675386) q[3];
rz(-0.00019970099566774213) q[3];
ry(-1.5705933820402693) q[4];
rz(-1.5729574847846) q[4];
ry(-1.5783293739306141) q[5];
rz(-0.011595931619350566) q[5];
ry(-3.140083756817011) q[6];
rz(-0.7050357607866913) q[6];
ry(-1.119469425951325) q[7];
rz(-1.0187824496866367) q[7];
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
ry(0.14350163696179252) q[0];
rz(-1.4742296465664424) q[0];
ry(3.141545497425456) q[1];
rz(2.100362207065983) q[1];
ry(1.5717842963356077) q[2];
rz(0.2494919783502754) q[2];
ry(2.3498795342477146) q[3];
rz(2.9198909657109837) q[3];
ry(-0.03598717047568556) q[4];
rz(-0.1647620935374829) q[4];
ry(0.9076969273090424) q[5];
rz(0.7876092093777103) q[5];
ry(-3.141537074472765) q[6];
rz(0.08542753198516315) q[6];
ry(-2.03169951217206e-06) q[7];
rz(-2.208259056127799) q[7];
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
ry(1.5707776267887088) q[0];
rz(-0.02465866227626691) q[0];
ry(0.00016376707938370316) q[1];
rz(-0.48866953591830947) q[1];
ry(-2.236669521063462e-05) q[2];
rz(1.3748399467958834) q[2];
ry(-3.140259776177057) q[3];
rz(1.3490868427042189) q[3];
ry(-5.142073853430166e-05) q[4];
rz(3.0837227831962606) q[4];
ry(-1.5783343881967173) q[5];
rz(1.5038541104318182) q[5];
ry(-1.571886360678004) q[6];
rz(-1.5511175407723106) q[6];
ry(-1.5708337336990894) q[7];
rz(-1.512293946648513) q[7];
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
ry(3.127754499574443) q[0];
rz(-0.7124038734088333) q[0];
ry(-3.141578252271998) q[1];
rz(-2.4913579659514418) q[1];
ry(3.141555323022308) q[2];
rz(2.506672143635775) q[2];
ry(-1.5708966303460565) q[3];
rz(-1.2622702283953162) q[3];
ry(-3.147603291205106e-05) q[4];
rz(-0.4610052807053169) q[4];
ry(0.00010552358736526034) q[5];
rz(1.9539800859887335) q[5];
ry(5.7326875764651675e-05) q[6];
rz(2.435673401266849) q[6];
ry(-7.034646738229554e-05) q[7];
rz(-1.3159725068051538) q[7];