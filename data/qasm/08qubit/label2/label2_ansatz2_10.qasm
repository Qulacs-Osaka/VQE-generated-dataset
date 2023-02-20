OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-3.141575197732225) q[0];
rz(-1.8940312865641133) q[0];
ry(-3.141342710619283) q[1];
rz(2.39344740718592) q[1];
ry(-3.0733606089066883) q[2];
rz(0.008473456794183817) q[2];
ry(3.1302422606039753) q[3];
rz(2.4320570263298866) q[3];
ry(-1.5705259919793224) q[4];
rz(3.067912359902362) q[4];
ry(1.5702581971189842) q[5];
rz(-0.0001998837689143329) q[5];
ry(6.133014556208849e-05) q[6];
rz(0.5442409361668582) q[6];
ry(-1.8480253506489524e-06) q[7];
rz(0.45041025363566034) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-3.1415564703840344) q[0];
rz(-2.095377550299986) q[0];
ry(3.215947386396027e-06) q[1];
rz(-2.4026092843707336) q[1];
ry(1.5698619413663997) q[2];
rz(1.570024621594099) q[2];
ry(-1.5702837598047656) q[3];
rz(1.4875985128192886) q[3];
ry(-2.4842007819906167) q[4];
rz(1.1596954484608006) q[4];
ry(0.9635681000353686) q[5];
rz(-2.762382827408911) q[5];
ry(3.1415766856350293) q[6];
rz(-2.2814699776164353) q[6];
ry(3.1415376663287073) q[7];
rz(0.13525287385477666) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(4.858157173703859e-05) q[0];
rz(3.0921157964651043) q[0];
ry(4.9047272169579734e-05) q[1];
rz(1.9497298894295056) q[1];
ry(-1.4874312044066673) q[2];
rz(1.6949281054134948) q[2];
ry(1.569947998173375) q[3];
rz(1.4331078242974264) q[3];
ry(3.1412341233544128) q[4];
rz(-1.091093104557821) q[4];
ry(-0.00017570719375984434) q[5];
rz(-2.753630900177955) q[5];
ry(-1.5712086921084927) q[6];
rz(-1.019206831862947) q[6];
ry(-1.5734468617904775) q[7];
rz(0.025341482369305446) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.571241224263466) q[0];
rz(3.1316414010057594) q[0];
ry(1.5680574770982745) q[1];
rz(3.0821560024807306) q[1];
ry(-1.5709931552401484) q[2];
rz(2.255722775313162) q[2];
ry(1.5776589372029157) q[3];
rz(3.141443306444661) q[3];
ry(-1.573543822170161) q[4];
rz(1.5410128705121382) q[4];
ry(-0.04808877922466214) q[5];
rz(1.636111551185701) q[5];
ry(-3.11051506048442) q[6];
rz(-2.5378501958528425) q[6];
ry(1.55206958269373) q[7];
rz(-1.6299334919360802) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.5674458509147158) q[0];
rz(-1.5692890402214754) q[0];
ry(2.144328695925524) q[1];
rz(-0.03261601709437063) q[1];
ry(0.00031399307595147974) q[2];
rz(0.8917326338992063) q[2];
ry(1.571362974307748) q[3];
rz(3.141061473251927) q[3];
ry(0.0003877990037606243) q[4];
rz(1.5975759099621014) q[4];
ry(1.570399655695851) q[5];
rz(-0.3679070861152854) q[5];
ry(1.640143675887698) q[6];
rz(2.4341677210626567) q[6];
ry(-1.6401002685730959) q[7];
rz(-3.101338342037882) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.23011477416226178) q[0];
rz(-1.582263101450093) q[0];
ry(1.5726878154263464) q[1];
rz(2.9116650763340473) q[1];
ry(1.5708356136764128) q[2];
rz(-0.0016553023936136668) q[2];
ry(1.5737475729076476) q[3];
rz(1.5722729888307176) q[3];
ry(-1.5702782645396403) q[4];
rz(1.5708187438245769) q[4];
ry(0.00040496263500612383) q[5];
rz(0.6577411374734831) q[5];
ry(-3.1415915292426972) q[6];
rz(-0.7110350580422446) q[6];
ry(3.1415913308795793) q[7];
rz(-3.0972541388534287) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.5719711921905513) q[0];
rz(-2.139273010485101) q[0];
ry(-1.571144440670146) q[1];
rz(1.6098534146587937) q[1];
ry(-1.5708689566868408) q[2];
rz(-0.3849786575568323) q[2];
ry(1.5224840050807957) q[3];
rz(-2.2488616576697624) q[3];
ry(1.572491052040113) q[4];
rz(-3.141580541377353) q[4];
ry(3.140527796883982) q[5];
rz(-2.851755438249505) q[5];
ry(1.5682065296026522) q[6];
rz(-3.141350842236432) q[6];
ry(-1.5733713443515454) q[7];
rz(-3.141346258843855) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(3.132836245364744) q[0];
rz(0.8414254651719596) q[0];
ry(-1.5702515043547463) q[1];
rz(-3.140111341615979) q[1];
ry(-0.0006691914830652124) q[2];
rz(-1.1860132889861783) q[2];
ry(0.0002084159739617064) q[3];
rz(-0.5275562984973606) q[3];
ry(1.5697463603663877) q[4];
rz(-2.997432879425776) q[4];
ry(1.5708893300815903) q[5];
rz(-0.11820597331438136) q[5];
ry(-1.570679976906944) q[6];
rz(-1.583174450005423) q[6];
ry(-1.5708084151851733) q[7];
rz(-1.5826808362585547) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.0015491816507342546) q[0];
rz(1.7170851987262385) q[0];
ry(-1.5693409455421468) q[1];
rz(1.5719667023358879) q[1];
ry(-1.5708916817493572) q[2];
rz(-1.571244608368573) q[2];
ry(3.141408846353979) q[3];
rz(0.40357556297620095) q[3];
ry(-3.0571819876793622) q[4];
rz(0.17196417169459277) q[4];
ry(0.0013749352271106659) q[5];
rz(-1.4760308936139133) q[5];
ry(2.4025377071448295) q[6];
rz(-0.028153027946718298) q[6];
ry(2.353799543946709) q[7];
rz(-0.24612854732556944) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.6216081812888774) q[0];
rz(1.5771856177364174) q[0];
ry(-1.5206398606705458) q[1];
rz(-1.56312820292088) q[1];
ry(1.5310316025814628) q[2];
rz(1.5724560930010987) q[2];
ry(0.04094852614140258) q[3];
rz(-2.7122217253147607) q[3];
ry(-1.4414380515042022) q[4];
rz(-0.014108475910528107) q[4];
ry(1.5708550166836115) q[5];
rz(-3.1415084469581043) q[5];
ry(-3.1377651285525534) q[6];
rz(-2.981574001167604) q[6];
ry(-3.1368576152410608) q[7];
rz(-2.7257879333943995) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.5378418477946045) q[0];
rz(-0.5863451932773275) q[0];
ry(1.6033838174534956) q[1];
rz(1.1658473792503292) q[1];
ry(-0.6285839027364304) q[2];
rz(-1.568417228023189) q[2];
ry(0.07381031356253208) q[3];
rz(-2.1636078792813995) q[3];
ry(1.6223305537978776) q[4];
rz(-1.5712231558774308) q[4];
ry(1.5707328294520377) q[5];
rz(-0.509070231243819) q[5];
ry(3.141454616142124) q[6];
rz(1.6460972333432202) q[6];
ry(-0.00015973707103356105) q[7];
rz(-2.0494810954208007) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.0005034936280373011) q[0];
rz(-2.1126773220214723) q[0];
ry(0.00022847558736684083) q[1];
rz(1.2474082469889205) q[1];
ry(1.5671701253275254) q[2];
rz(-2.9709327037593436) q[2];
ry(-0.0024112613112868796) q[3];
rz(-0.14378217748130684) q[3];
ry(-1.583553173057136) q[4];
rz(2.018868288144132) q[4];
ry(0.0006597977910809406) q[5];
rz(-1.1012066538563765) q[5];
ry(-3.11166892301476) q[6];
rz(1.6436069053370588) q[6];
ry(-1.6020779968117627) q[7];
rz(1.57248338629371) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(3.546035873291942e-05) q[0];
rz(1.8960006255777841) q[0];
ry(-3.141563002354882) q[1];
rz(1.6114126780277616) q[1];
ry(2.2440263168554964e-05) q[2];
rz(2.216417575092945) q[2];
ry(3.533297283891131e-05) q[3];
rz(1.0673713561948928) q[3];
ry(0.0011160855673834381) q[4];
rz(-2.461690315010995) q[4];
ry(-3.1415409006781303) q[5];
rz(-0.6178948494581759) q[5];
ry(3.117175250452245) q[6];
rz(-1.9646398172800605) q[6];
ry(-1.5377010034869727) q[7];
rz(1.1260260204858006) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.011780805398856) q[0];
rz(-2.6130127702223978) q[0];
ry(2.0124093520953465) q[1];
rz(-2.613024458127196) q[1];
ry(-1.1684466070928519) q[2];
rz(-2.649451484972202) q[2];
ry(-1.1177336167188805) q[3];
rz(-2.608303605416225) q[3];
ry(-1.6986693186600732) q[4];
rz(-1.101195018992458) q[4];
ry(-0.8909553070477215) q[5];
rz(-0.651451553418473) q[5];
ry(-0.8902121058529538) q[6];
rz(-0.6512073102229806) q[6];
ry(-2.370278308365088) q[7];
rz(-1.2378846427084245) q[7];