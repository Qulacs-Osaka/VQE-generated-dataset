OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.5804590363327637) q[0];
rz(-0.13744329131141075) q[0];
ry(1.5708268770943663) q[1];
rz(-3.1414790689692786) q[1];
ry(-0.3933205227286205) q[2];
rz(3.098174790352298) q[2];
ry(1.949426429626075) q[3];
rz(-0.658194983767693) q[3];
ry(-1.5639715567135692) q[4];
rz(-2.061894950162373) q[4];
ry(1.4668223993041867) q[5];
rz(1.6899752256878222) q[5];
ry(0.03449354667601501) q[6];
rz(-0.412764956376321) q[6];
ry(-3.1400840092536586) q[7];
rz(-0.959407100793775) q[7];
ry(1.570965672009769) q[8];
rz(0.0013235802478475454) q[8];
ry(-1.5706941134455055) q[9];
rz(0.2213380720362803) q[9];
ry(-0.06373346280159708) q[10];
rz(1.5281699509970839) q[10];
ry(-1.1155456475892507) q[11];
rz(-2.6886810470770257) q[11];
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
ry(1.7148963590282715) q[0];
rz(1.530686814819747) q[0];
ry(-2.811327826867303) q[1];
rz(0.13302512204575057) q[1];
ry(3.1172105423025194) q[2];
rz(1.369106147298159) q[2];
ry(3.1406893766174284) q[3];
rz(-2.167352183027007) q[3];
ry(-0.0013038706818132978) q[4];
rz(-2.658724520171853) q[4];
ry(3.1409923806286058) q[5];
rz(0.10454387381605458) q[5];
ry(-0.0010724614129440369) q[6];
rz(-1.724889338118357) q[6];
ry(-3.141246721985544) q[7];
rz(-1.4115467259154073) q[7];
ry(-1.0752848535471795) q[8];
rz(1.5701150129257073) q[8];
ry(-0.000861811859756459) q[9];
rz(1.3494372905381213) q[9];
ry(-1.6378734880790635) q[10];
rz(2.024082396784341) q[10];
ry(-0.0236669603980167) q[11];
rz(1.5628600141731133) q[11];
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
ry(1.869282791283041) q[0];
rz(1.5654904783744312) q[0];
ry(-9.531256457862012e-06) q[1];
rz(-0.13348654351794664) q[1];
ry(2.505924567675851) q[2];
rz(2.10343287417583) q[2];
ry(2.297275770107391) q[3];
rz(1.582209186069571) q[3];
ry(2.3971848311401276) q[4];
rz(-1.567806189209558) q[4];
ry(0.44905594197129733) q[5];
rz(2.9001963429915234) q[5];
ry(0.01139601815259361) q[6];
rz(2.149848515789848) q[6];
ry(0.00023471922631745912) q[7];
rz(0.6848571014843141) q[7];
ry(0.2834152279604636) q[8];
rz(-2.140145970731048) q[8];
ry(2.0171150595246683) q[9];
rz(2.6349215991114736) q[9];
ry(1.5526623338032515) q[10];
rz(0.006613903597341303) q[10];
ry(-3.1341766579392885) q[11];
rz(2.0158329332958953) q[11];
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
ry(-2.6548447799635397) q[0];
rz(-1.4903737475916543) q[0];
ry(2.927874181724205) q[1];
rz(-1.571376707126853) q[1];
ry(-0.14947192002052567) q[2];
rz(-2.960329391261671) q[2];
ry(-1.422941973950843) q[3];
rz(-1.397891526333244) q[3];
ry(1.8750806635529402) q[4];
rz(-1.5626872078676) q[4];
ry(-2.970054265839612) q[5];
rz(-2.137601575325756) q[5];
ry(-1.5695794741004312) q[6];
rz(-1.563374079762295) q[6];
ry(-1.5709780979582837) q[7];
rz(-3.0707250688542795) q[7];
ry(-3.141068955213839) q[8];
rz(2.354228575996366) q[8];
ry(3.1415606094239865) q[9];
rz(-0.5277187873240414) q[9];
ry(-0.48725707764504905) q[10];
rz(0.5371231579263464) q[10];
ry(1.5679835425375832) q[11];
rz(0.11814450114549775) q[11];
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
ry(3.139590124351864) q[0];
rz(1.6329238837416309) q[0];
ry(-1.4584207493859722) q[1];
rz(-0.7474856556301902) q[1];
ry(-0.0010117339142210469) q[2];
rz(2.5504538840903273) q[2];
ry(-3.255783385824884e-05) q[3];
rz(-1.3188246736609044) q[3];
ry(1.356103050911496) q[4];
rz(0.45663275196765274) q[4];
ry(-3.1386024121910188) q[5];
rz(-1.8823135291685467) q[5];
ry(-1.5705544976379509) q[6];
rz(0.025248960463010173) q[6];
ry(3.1373609093156345) q[7];
rz(0.07361470188950216) q[7];
ry(-4.265220806143333e-05) q[8];
rz(-2.923273824039993) q[8];
ry(-0.09972873572650176) q[9];
rz(-1.5500828471583048) q[9];
ry(-0.0005469451681123196) q[10];
rz(-0.5283135382255911) q[10];
ry(3.9666532226867446e-05) q[11];
rz(-0.10925136271055802) q[11];
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
ry(3.136469223519244) q[0];
rz(-1.6362045296929075) q[0];
ry(-3.1415761181324577) q[1];
rz(-0.7476370610323864) q[1];
ry(1.569897265209078) q[2];
rz(2.233021937578738) q[2];
ry(-3.137355354294154) q[3];
rz(-1.592272562996934) q[3];
ry(-0.004172794853587014) q[4];
rz(2.6846168086580406) q[4];
ry(-1.5707884406090882) q[5];
rz(-0.0005737432256732434) q[5];
ry(0.0038779897203768954) q[6];
rz(-0.9465949026620537) q[6];
ry(2.391957396020325) q[7];
rz(-0.7557014744899333) q[7];
ry(1.8570728214475798) q[8];
rz(0.0004275568917576153) q[8];
ry(0.28526856953476987) q[9];
rz(1.571196374436832) q[9];
ry(0.005370850909306668) q[10];
rz(-0.991807387536599) q[10];
ry(1.5730994329360275) q[11];
rz(0.0003817875173093654) q[11];
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
ry(3.1398038903316254) q[0];
rz(-2.023332357858509) q[0];
ry(-0.7626540984163075) q[1];
rz(1.5750356062229336) q[1];
ry(0.0008421830620060078) q[2];
rz(-2.233203636396622) q[2];
ry(1.566215063529536) q[3];
rz(-2.780500596381505) q[3];
ry(-1.1732733468951553) q[4];
rz(-0.39356513948160193) q[4];
ry(3.1399890127067915) q[5];
rz(-1.9808833648986983) q[5];
ry(3.141182049953831) q[6];
rz(-2.48818247219021) q[6];
ry(3.1410322707641978) q[7];
rz(1.1310258501453143) q[7];
ry(1.5702466160456563) q[8];
rz(-3.0547813134976964) q[8];
ry(-2.269378290670451) q[9];
rz(-1.593440649538716) q[9];
ry(0.0008531323335478902) q[10];
rz(-2.179737205321417) q[10];
ry(1.5714844651799031) q[11];
rz(-1.602043954234803) q[11];
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
ry(-1.5662109182210937) q[0];
rz(0.6299927417227638) q[0];
ry(-3.023776428945625) q[1];
rz(-1.8841631714348255) q[1];
ry(1.570385662279799) q[2];
rz(-1.5707193275270561) q[2];
ry(1.831767915106544e-05) q[3];
rz(1.7009465706551825) q[3];
ry(-1.5699763217247258) q[4];
rz(-1.4904410199762648) q[4];
ry(4.8468123051925716e-05) q[5];
rz(0.01760362216947886) q[5];
ry(-1.5708529399867812) q[6];
rz(-2.6244154982453445) q[6];
ry(-1.270220692629863e-05) q[7];
rz(-2.9486273052727747) q[7];
ry(1.717487911997574) q[8];
rz(2.9296348387766287) q[8];
ry(0.11226976757058882) q[9];
rz(-3.0430032033852017) q[9];
ry(3.139590628373209) q[10];
rz(1.6710550001468085) q[10];
ry(-0.00013189997454698954) q[11];
rz(0.32341909103418914) q[11];
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
ry(0.00011280641195643426) q[0];
rz(2.513611611695323) q[0];
ry(-0.0006442747384243219) q[1];
rz(1.8887044980176066) q[1];
ry(1.570220594558017) q[2];
rz(2.815990185721762e-05) q[2];
ry(0.0015000856179936863) q[3];
rz(0.6029073563731083) q[3];
ry(0.0003885142004067532) q[4];
rz(-1.653528584560794) q[4];
ry(-0.00018281388618923697) q[5];
rz(-2.752887199456003) q[5];
ry(-3.1394567386945678) q[6];
rz(-2.526280519316043) q[6];
ry(0.003160524460731917) q[7];
rz(-2.0821474889011893) q[7];
ry(1.5707958586737887) q[8];
rz(1.0778449808665913) q[8];
ry(-3.141585491149052) q[9];
rz(0.07544375555621041) q[9];
ry(-3.1405220972989545) q[10];
rz(1.717696594747089) q[10];
ry(0.01114144664113148) q[11];
rz(-2.515458463498146) q[11];
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
ry(-1.5730127160682243) q[0];
rz(-3.141304072571441) q[0];
ry(-2.7996236433417216) q[1];
rz(-1.5703333375106532) q[1];
ry(1.5709103868959788) q[2];
rz(-2.1837298216129) q[2];
ry(0.016709683389516096) q[3];
rz(-0.9762077236143399) q[3];
ry(1.5704924681323968) q[4];
rz(6.212495268176356e-05) q[4];
ry(-1.5745923210851687) q[5];
rz(-1.3832166833876602) q[5];
ry(-3.141188611130447) q[6];
rz(2.8930564765224625) q[6];
ry(-1.568801882490809) q[7];
rz(-1.1766373907490733) q[7];
ry(0.00036734331793208947) q[8];
rz(-0.7270486465658474) q[8];
ry(-2.7806611556130925) q[9];
rz(1.570119772832844) q[9];
ry(-1.560438697058225) q[10];
rz(-0.000276130187492285) q[10];
ry(3.1397415777391924) q[11];
rz(2.8696194744728984) q[11];
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
ry(-1.8184459549939493) q[0];
rz(-2.669793380586621) q[0];
ry(2.0308623605609086) q[1];
rz(-1.606639490253343) q[1];
ry(1.5709676344906784) q[2];
rz(-1.5709304760162253) q[2];
ry(0.0001430585043170169) q[3];
rz(-2.8099061033178976) q[3];
ry(1.5086906057560823) q[4];
rz(1.5387136409223094) q[4];
ry(-3.700421285689182e-05) q[5];
rz(-0.732361025578236) q[5];
ry(-0.00016134576419979393) q[6];
rz(-2.1733610139672215) q[6];
ry(-3.14150779152838) q[7];
rz(-1.9115941957982856) q[7];
ry(0.009686431776310124) q[8];
rz(-1.9218306340922706) q[8];
ry(-2.0147045356994644) q[9];
rz(-1.5707764042270493) q[9];
ry(1.5705930806763837) q[10];
rz(-0.016235538505851743) q[10];
ry(-3.314293758815979e-05) q[11];
rz(-3.0777958746380603) q[11];
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
ry(0.0025117669385394947) q[0];
rz(1.0992111109602614) q[0];
ry(0.007864638724720858) q[1];
rz(0.8557458774178699) q[1];
ry(-1.5614427085780591) q[2];
rz(1.5702635394678655) q[2];
ry(0.0004574894563011966) q[3];
rz(-1.7480317750956942) q[3];
ry(0.016149825703792867) q[4];
rz(-3.1094514923571577) q[4];
ry(3.1409859764785293) q[5];
rz(-0.1018347036303169) q[5];
ry(0.0007887136084590054) q[6];
rz(0.9493765469195611) q[6];
ry(3.140969447021713) q[7];
rz(-0.04467618811496976) q[7];
ry(-1.6647564658406324) q[8];
rz(3.1413220138448823) q[8];
ry(-1.5794640767741945) q[9];
rz(3.1415207630655058) q[9];
ry(-1.570834109887885) q[10];
rz(-2.586828140251752) q[10];
ry(3.1397567107410973) q[11];
rz(0.4779183853475857) q[11];
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
ry(-1.5702806886586231) q[0];
rz(1.4278397500568598) q[0];
ry(2.1646909632961893e-05) q[1];
rz(-0.4082649849829198) q[1];
ry(1.570756770287981) q[2];
rz(-1.292529724699652) q[2];
ry(-3.140739677814862) q[3];
rz(1.8422420558149408) q[3];
ry(1.5705231445899723) q[4];
rz(3.0133719378099335) q[4];
ry(3.141055017689398) q[5];
rz(-2.6985314976678114) q[5];
ry(1.5706864847910218) q[6];
rz(2.147928244782328) q[6];
ry(0.0038807014602376334) q[7];
rz(0.8780094723334524) q[7];
ry(-1.5714516258380218) q[8];
rz(0.8957147271171557) q[8];
ry(-1.5707440216620308) q[9];
rz(-0.3662011054277495) q[9];
ry(0.0006423660807426401) q[10];
rz(-1.4025293449003913) q[10];
ry(-3.1061077475378647) q[11];
rz(0.0347451295885845) q[11];
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
ry(-4.286969851996503e-05) q[0];
rz(0.14290931334160015) q[0];
ry(-3.141388463872019) q[1];
rz(-2.729991185005298) q[1];
ry(6.047106583183974e-06) q[2];
rz(1.2925286883578346) q[2];
ry(-1.5709147895868882) q[3];
rz(3.790289200252409e-05) q[3];
ry(-2.190654736068887e-06) q[4];
rz(1.6989426929821025) q[4];
ry(-1.5709376113624218) q[5];
rz(-1.5837528511953658e-05) q[5];
ry(-9.004791587220906e-06) q[6];
rz(-0.5771859902648133) q[6];
ry(-1.5709331405800775) q[7];
rz(3.141571941353992) q[7];
ry(3.140707834701256) q[8];
rz(-0.6749070148545018) q[8];
ry(-3.141557455205803) q[9];
rz(0.2111042049941556) q[9];
ry(3.1395359959267783) q[10];
rz(0.7229236597679418) q[10];
ry(-1.5709409749752585) q[11];
rz(0.00035178672602764754) q[11];
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
ry(1.571004020029571) q[0];
rz(2.7854143064372643) q[0];
ry(-1.570760644555505) q[1];
rz(-1.4569246802374562) q[1];
ry(1.5707901097484225) q[2];
rz(-0.3565388787273225) q[2];
ry(-1.5708245682849287) q[3];
rz(-3.0283601524312433) q[3];
ry(1.5709823569484733) q[4];
rz(-0.35589890927919515) q[4];
ry(-1.570790314199594) q[5];
rz(-1.4579038356884055) q[5];
ry(1.5708036530389062) q[6];
rz(1.2146880842182903) q[6];
ry(1.5708024651539993) q[7];
rz(0.1126451182783817) q[7];
ry(1.5713796529712178) q[8];
rz(2.7849306102985243) q[8];
ry(-3.141528165861776) q[9];
rz(-2.452549532304525) q[9];
ry(1.5709024175033637) q[10];
rz(1.2137529019232791) q[10];
ry(1.5707064671786748) q[11];
rz(0.11235088634365313) q[11];