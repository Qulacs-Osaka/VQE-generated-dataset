OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(2.191236377359699) q[0];
rz(1.4016943934792891) q[0];
ry(3.1415199682576596) q[1];
rz(-1.4443351192989151) q[1];
ry(-1.5708001985952604) q[2];
rz(1.5707891557964537) q[2];
ry(-1.5703137440322228) q[3];
rz(-2.4202137679943694) q[3];
ry(-1.5707733445328427) q[4];
rz(1.5709182477027386) q[4];
ry(-2.8636870264373124) q[5];
rz(-1.5707681803788942) q[5];
ry(1.5708054948966819) q[6];
rz(1.5707923443090481) q[6];
ry(1.568818321334439) q[7];
rz(1.5706057391185766) q[7];
ry(-1.5707872106006795) q[8];
rz(-1.5708261076860883) q[8];
ry(-1.5705151277547251) q[9];
rz(0.013798564580995711) q[9];
ry(-1.5707978185625182) q[10];
rz(-1.570793170748007) q[10];
ry(-1.5199435149732745) q[11];
rz(2.384611362948874e-05) q[11];
ry(-1.5708026727918556) q[12];
rz(1.5707788489727417) q[12];
ry(-1.5705846671381851) q[13];
rz(-0.11998521994137126) q[13];
ry(1.570788467423438) q[14];
rz(-1.5707064034352767) q[14];
ry(2.1026246877984027) q[15];
rz(-3.1414755251905446) q[15];
ry(-1.5707719975405299) q[16];
rz(1.5707918288600127) q[16];
ry(1.5707948959274933) q[17];
rz(-2.9399912628281606) q[17];
ry(-1.5707983344395589) q[18];
rz(-1.5714764809431077) q[18];
ry(-1.097685058248521) q[19];
rz(5.838077781597861e-05) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.8020480512611323) q[0];
rz(-0.8010345377980476) q[0];
ry(-2.383851606191188) q[1];
rz(-1.57066437425277) q[1];
ry(1.833301290844288) q[2];
rz(5.490970088395386e-06) q[2];
ry(-1.5712745034309554) q[3];
rz(3.141583251081231) q[3];
ry(-3.0167999270082317) q[4];
rz(-3.141470710720129) q[4];
ry(0.4092262488133658) q[5];
rz(1.5708120030622594) q[5];
ry(-1.5708913031454692) q[6];
rz(-2.9850442975352474) q[6];
ry(-1.5707185098012952) q[7];
rz(0.014260452288736403) q[7];
ry(0.3402202834141068) q[8];
rz(1.0427488982232648e-05) q[8];
ry(-1.5721914888688056) q[9];
rz(3.038691671054095) q[9];
ry(2.8378457114089928) q[10];
rz(1.2570069753081723e-05) q[10];
ry(1.5708833582762445) q[11];
rz(0.011222038983498273) q[11];
ry(-0.3067211832966157) q[12];
rz(-3.1415597026654503) q[12];
ry(-1.5740992543198216) q[13];
rz(-0.10292687715601012) q[13];
ry(-2.7980272939183286) q[14];
rz(5.97462634775202e-05) q[14];
ry(1.5709297243544962) q[15];
rz(-3.132685675974771) q[15];
ry(0.3440021993518334) q[16];
rz(3.1415898280549635) q[16];
ry(1.5699540509210819) q[17];
rz(-2.302517311906726) q[17];
ry(-2.8971942401400295) q[18];
rz(-1.5706310981153795) q[18];
ry(1.570796434865529) q[19];
rz(-1.5700205724946645) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-3.1414379782523296) q[0];
rz(-1.7316983952237837) q[0];
ry(1.5707977629773902) q[1];
rz(-1.5707630352379676) q[1];
ry(1.5707906647065417) q[2];
rz(1.570814820429522) q[2];
ry(-1.5713005837338851) q[3];
rz(1.5708076521256389) q[3];
ry(-1.5707852826917976) q[4];
rz(-1.5708125211882802) q[4];
ry(-1.570815113411635) q[5];
rz(1.5713572052505809) q[5];
ry(-0.00046569166563958936) q[6];
rz(1.4142502878986045) q[6];
ry(-1.572730160505977) q[7];
rz(0.0185164215078445) q[7];
ry(-2.7880330237384685) q[8];
rz(-1.5708074032992698) q[8];
ry(3.1412912393245698) q[9];
rz(1.5782080673342107) q[9];
ry(-0.3005969533891248) q[10];
rz(-1.5708020013154929) q[10];
ry(1.6216157530229827) q[11];
rz(-0.242315630779518) q[11];
ry(0.3068860001488289) q[12];
rz(1.5707812369180556) q[12];
ry(-3.1413538725188825) q[13];
rz(1.5797010529661184) q[13];
ry(0.3421688996603962) q[14];
rz(-1.5707845067390567) q[14];
ry(-1.0389675319229026) q[15];
rz(-3.096629144253759) q[15];
ry(-2.846498958660139) q[16];
rz(1.5707911409672695) q[16];
ry(-3.1415909779961986) q[17];
rz(2.369126613092684) q[17];
ry(-2.3052226954928705) q[18];
rz(-1.5708058085915888) q[18];
ry(1.571142925147397) q[19];
rz(2.8331230933980223) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.5708301019907627) q[0];
rz(2.2448412937939524) q[0];
ry(1.6993524596874625) q[1];
rz(3.072679420191552) q[1];
ry(1.5707970437948708) q[2];
rz(-2.467540720324944) q[2];
ry(-1.570820877104595) q[3];
rz(1.5020932219787735) q[3];
ry(1.5707954319466766) q[4];
rz(0.6738357041027976) q[4];
ry(-1.6908725412847332) q[5];
rz(3.072495700253421) q[5];
ry(-1.5708100922296415) q[6];
rz(-0.8969767467003409) q[6];
ry(-3.141506751134002) q[7];
rz(3.090805642304406) q[7];
ry(1.5708025175385418) q[8];
rz(-2.4670725967175415) q[8];
ry(1.5706342540975504) q[9];
rz(-1.6386444452837168) q[9];
ry(-1.5707925711008022) q[10];
rz(-0.8948360338475164) q[10];
ry(-9.972747694675377e-05) q[11];
rz(-2.96862210160817) q[11];
ry(1.5707949255781402) q[12];
rz(0.6599317394647802) q[12];
ry(-1.571199702296456) q[13];
rz(-1.6436929119043846) q[13];
ry(-1.5707422122021737) q[14];
rz(-0.8353384017566148) q[14];
ry(-3.141430370797734) q[15];
rz(-0.02113044044907264) q[15];
ry(1.5707959142923178) q[16];
rz(-2.5856731236781574) q[16];
ry(1.5707170888691326) q[17];
rz(1.5003811663465496) q[17];
ry(1.5707729273493038) q[18];
rz(-0.5204726147920017) q[18];
ry(-3.141529446981368) q[19];
rz(0.10690129866327956) q[19];