OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.050133857361661) q[0];
rz(1.867145453950363) q[0];
ry(3.1411504917175264) q[1];
rz(-0.2942684058628772) q[1];
ry(1.5697346996479802) q[2];
rz(-1.652464705789204) q[2];
ry(-1.4416832729475908) q[3];
rz(0.037294949139389466) q[3];
ry(-0.0002420157236464604) q[4];
rz(-0.050894527709935744) q[4];
ry(0.00038673565484850493) q[5];
rz(0.48809103390754954) q[5];
ry(-1.5699031148061673) q[6];
rz(2.239562941416864) q[6];
ry(1.5708536375398827) q[7];
rz(2.8216622180666517) q[7];
ry(-0.00021041764381152052) q[8];
rz(-1.962720731632879) q[8];
ry(3.141553413544557) q[9];
rz(2.3725777580440823) q[9];
ry(-3.1401145000864865) q[10];
rz(-0.469279635712386) q[10];
ry(0.0029980928970204) q[11];
rz(2.9692134004767463) q[11];
ry(-1.5394194138949544) q[12];
rz(1.78984075633305) q[12];
ry(1.5364448997173774) q[13];
rz(-3.1280684092476254) q[13];
ry(0.001054976568537036) q[14];
rz(3.0739143033314393) q[14];
ry(-6.122424107246616e-05) q[15];
rz(-2.3539088380740707) q[15];
ry(1.5709017516195996) q[16];
rz(2.4547705281149605) q[16];
ry(-1.5710498535372213) q[17];
rz(1.4424299108147074) q[17];
ry(0.027399113335398653) q[18];
rz(-0.04837340140136673) q[18];
ry(-3.068134474298912) q[19];
rz(-1.5766183045431132) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-3.1282733335528463) q[0];
rz(-2.9659880086567503) q[0];
ry(-0.008852038481612645) q[1];
rz(-1.2255668639281432) q[1];
ry(0.14006625184032906) q[2];
rz(-2.6947176362342704) q[2];
ry(-1.654306003174895) q[3];
rz(-1.9245240127400653) q[3];
ry(-1.7738575878755458) q[4];
rz(2.1761959023910853) q[4];
ry(1.475192054579617) q[5];
rz(1.3604208489052256) q[5];
ry(-2.980772711458605) q[6];
rz(2.4455377277122863) q[6];
ry(1.7539256762280182) q[7];
rz(1.7007091943253183) q[7];
ry(3.1390025294880415) q[8];
rz(2.455222040970756) q[8];
ry(0.012952169388827704) q[9];
rz(-2.2330719824025893) q[9];
ry(-1.6606773828614996) q[10];
rz(3.1114516096245635) q[10];
ry(-1.4700438870753567) q[11];
rz(-1.2088744009885406) q[11];
ry(3.1290904319200368) q[12];
rz(-1.5084801995418007) q[12];
ry(-1.662830731862621) q[13];
rz(3.0493171868499886) q[13];
ry(-2.914507461524795) q[14];
rz(3.1180702163362826) q[14];
ry(-1.6441811038471383) q[15];
rz(-0.07456313390265468) q[15];
ry(-1.6757306651685253) q[16];
rz(3.0405485419399194) q[16];
ry(-2.3871756747167807) q[17];
rz(1.370260966726966) q[17];
ry(3.0359440773396145) q[18];
rz(-1.6177529554930141) q[18];
ry(-1.5651638508353969) q[19];
rz(1.4978474186625617) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5349329475559854) q[0];
rz(-1.726721513881221) q[0];
ry(-1.5735510389595895) q[1];
rz(0.0040143508904151295) q[1];
ry(-1.570772361429536) q[2];
rz(-1.5699078758592027) q[2];
ry(-1.570746581339241) q[3];
rz(-0.0026431825360520023) q[3];
ry(3.1275900250986415) q[4];
rz(3.1133080396596062) q[4];
ry(-0.003033572134754081) q[5];
rz(-2.975026755724128) q[5];
ry(0.02338148831889164) q[6];
rz(-1.422826343179131) q[6];
ry(3.095465080457916) q[7];
rz(-2.559830077329273) q[7];
ry(0.31742893205142386) q[8];
rz(3.1371957822020256) q[8];
ry(-4.892073386785244e-05) q[9];
rz(-1.363012356701334) q[9];
ry(-3.1110095749908275) q[10];
rz(-0.0381146259851901) q[10];
ry(0.008288485494994679) q[11];
rz(1.209957934264892) q[11];
ry(-3.049854679738678) q[12];
rz(2.9737569242826694) q[12];
ry(2.985956561863765) q[13];
rz(-0.1052038877269279) q[13];
ry(-2.5436081330189575) q[14];
rz(0.01180545834873833) q[14];
ry(-0.0011921147141880084) q[15];
rz(-1.493882483461097) q[15];
ry(0.8994720306620323) q[16];
rz(-1.68726028186103) q[16];
ry(0.8983729145641953) q[17];
rz(-1.6929482126607325) q[17];
ry(-2.488855022955062) q[18];
rz(-1.5538187952013196) q[18];
ry(0.0008510101249861179) q[19];
rz(-0.729662990795676) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.01222733053787373) q[0];
rz(1.842556199908596) q[0];
ry(-1.5726998899448856) q[1];
rz(-3.024369872901535) q[1];
ry(1.8191318687885643) q[2];
rz(-1.8310584562535244) q[2];
ry(-1.5706244507470704) q[3];
rz(-2.8962836398383356) q[3];
ry(-2.8167628896217893) q[4];
rz(0.8532423237398896) q[4];
ry(-1.866924467246436) q[5];
rz(0.704828349750473) q[5];
ry(1.5134701403304565) q[6];
rz(-1.249129197616056) q[6];
ry(-1.3837038891764588) q[7];
rz(-0.16612308431400066) q[7];
ry(-1.5239011611308708) q[8];
rz(-1.5681207010942417) q[8];
ry(1.5840261500210355) q[9];
rz(1.5702197976853107) q[9];
ry(1.4548584176780146) q[10];
rz(-1.567949007603901) q[10];
ry(1.6322450741352714) q[11];
rz(1.571287596011579) q[11];
ry(0.9695305104126841) q[12];
rz(-3.12439356180054) q[12];
ry(0.9970735857109004) q[13];
rz(0.01809133067077221) q[13];
ry(-1.2461672491181208) q[14];
rz(-0.006830970253164814) q[14];
ry(-1.5667885691975272) q[15];
rz(2.1761461022431137) q[15];
ry(-2.002213757264996) q[16];
rz(1.1687874292714797) q[16];
ry(-2.1011061610897896) q[17];
rz(1.3863353726014083) q[17];
ry(-0.797682475720757) q[18];
rz(-0.015802658716452346) q[18];
ry(-3.1294844107625357) q[19];
rz(-0.38611383409531447) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5708370505793676) q[0];
rz(-0.052433127886689156) q[0];
ry(-1.5708583508283107) q[1];
rz(3.0890391505518022) q[1];
ry(-3.11385641909051) q[2];
rz(-0.20662179734055638) q[2];
ry(-1.5696433026799084) q[3];
rz(1.1814026005288172) q[3];
ry(0.19533550685231713) q[4];
rz(1.5157507223463074) q[4];
ry(-0.017772623337643286) q[5];
rz(-0.4472226046804658) q[5];
ry(0.0001314304387583569) q[6];
rz(0.6505310970031741) q[6];
ry(-6.640653495913031e-05) q[7];
rz(-0.5840996638862487) q[7];
ry(1.5656677031088122) q[8];
rz(-0.5552363971140979) q[8];
ry(1.5699542219643803) q[9];
rz(2.515101209479284) q[9];
ry(-1.5709986625385706) q[10];
rz(-0.41311849457430583) q[10];
ry(-1.5708546060735173) q[11];
rz(1.0371703348142356) q[11];
ry(-1.5700588329080007) q[12];
rz(-2.236643701029) q[12];
ry(1.569080569793151) q[13];
rz(-2.1882400310616137) q[13];
ry(1.5727350008786187) q[14];
rz(-1.252235936464845) q[14];
ry(3.1382010768406774) q[15];
rz(-2.2904014550736007) q[15];
ry(-0.0006312149496688741) q[16];
rz(-1.2805250826373673) q[16];
ry(3.1410857755182704) q[17];
rz(3.049551749584106) q[17];
ry(-1.5725321480312888) q[18];
rz(-3.0899569784072765) q[18];
ry(0.0009343842934805835) q[19];
rz(-0.353556566438798) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5677126237806787) q[0];
rz(-1.9808231333433497) q[0];
ry(1.5881775578220685) q[1];
rz(1.159772854086287) q[1];
ry(1.546481959693761) q[2];
rz(-1.985600279011984) q[2];
ry(-0.1428147703451046) q[3];
rz(-1.5987210174245494) q[3];
ry(1.8737992946758313) q[4];
rz(-0.533770443716838) q[4];
ry(-2.813947965515719) q[5];
rz(-0.16005944921882911) q[5];
ry(-1.3177597502942304) q[6];
rz(1.1686395548096753) q[6];
ry(1.3172840805896406) q[7];
rz(-1.972837219843215) q[7];
ry(-0.6133504804828105) q[8];
rz(-3.066442124707716) q[8];
ry(2.7576588453047783) q[9];
rz(-2.627957301757742) q[9];
ry(-1.3079444309405517) q[10];
rz(2.865255166970278) q[10];
ry(-0.46839238867090843) q[11];
rz(-1.3755579369391802) q[11];
ry(1.9759794063770668) q[12];
rz(3.1115131704536885) q[12];
ry(-1.984033199416583) q[13];
rz(-0.03413386421716048) q[13];
ry(-0.6022068773965397) q[14];
rz(1.295943487825836) q[14];
ry(0.13944200989532085) q[15];
rz(2.998874390432746) q[15];
ry(-0.03463467908854361) q[16];
rz(1.7749156164521238) q[16];
ry(3.1384878831183007) q[17];
rz(0.18233191914993263) q[17];
ry(2.4905316839627987) q[18];
rz(0.17462695741101175) q[18];
ry(0.6155332601977908) q[19];
rz(1.7181177559070555) q[19];