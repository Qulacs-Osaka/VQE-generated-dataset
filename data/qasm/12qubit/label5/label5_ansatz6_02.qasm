OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.9343117523532039) q[0];
ry(1.0372469394808728) q[1];
cx q[0],q[1];
ry(2.678146638966226) q[0];
ry(-3.1188374416307196) q[1];
cx q[0],q[1];
ry(-0.19216328385729753) q[1];
ry(-1.5273324738771639) q[2];
cx q[1],q[2];
ry(2.4240122892345624) q[1];
ry(1.1495398397722694) q[2];
cx q[1],q[2];
ry(-2.162824559654356) q[2];
ry(-1.0868663864623553) q[3];
cx q[2],q[3];
ry(2.9308686142107683) q[2];
ry(1.040453893178181) q[3];
cx q[2],q[3];
ry(0.17948469542204748) q[3];
ry(-2.214309652849393) q[4];
cx q[3],q[4];
ry(-2.9948487124974865) q[3];
ry(0.0007227217548306797) q[4];
cx q[3],q[4];
ry(0.6538800391404145) q[4];
ry(-2.8267491256562987) q[5];
cx q[4],q[5];
ry(0.6937842548875242) q[4];
ry(-0.18431185339532813) q[5];
cx q[4],q[5];
ry(-0.17284646221683303) q[5];
ry(2.9424166523410924) q[6];
cx q[5],q[6];
ry(-1.529528618535753) q[5];
ry(2.905866069171629) q[6];
cx q[5],q[6];
ry(1.533738447883552) q[6];
ry(-2.7285486025743033) q[7];
cx q[6],q[7];
ry(-1.229829676778193) q[6];
ry(-3.140031490777638) q[7];
cx q[6],q[7];
ry(0.8976711784500371) q[7];
ry(3.136816612716819) q[8];
cx q[7],q[8];
ry(1.5966541500650866) q[7];
ry(-1.5357007839078864) q[8];
cx q[7],q[8];
ry(1.8260232017147526) q[8];
ry(-3.0206164784953735) q[9];
cx q[8],q[9];
ry(1.5293819473352714) q[8];
ry(1.5755231488651766) q[9];
cx q[8],q[9];
ry(-2.155621517468412) q[9];
ry(0.11722968008248211) q[10];
cx q[9],q[10];
ry(-2.5253065387722677) q[9];
ry(2.556307796012025) q[10];
cx q[9],q[10];
ry(0.09844755020444837) q[10];
ry(-3.000292953787556) q[11];
cx q[10],q[11];
ry(0.9615842412549771) q[10];
ry(-0.005241608507359885) q[11];
cx q[10],q[11];
ry(-0.6349284653127601) q[0];
ry(-2.8994264330733652) q[1];
cx q[0],q[1];
ry(-1.7658801876428312) q[0];
ry(2.882335539774891) q[1];
cx q[0],q[1];
ry(-2.449507456393171) q[1];
ry(-1.976473610655266) q[2];
cx q[1],q[2];
ry(-0.763128034582131) q[1];
ry(1.2898180588772865) q[2];
cx q[1],q[2];
ry(2.969411873643663) q[2];
ry(-1.1716047918076076) q[3];
cx q[2],q[3];
ry(-3.0189438006781626) q[2];
ry(-1.182731911477708) q[3];
cx q[2],q[3];
ry(1.3236869391940322) q[3];
ry(-2.780875469244227) q[4];
cx q[3],q[4];
ry(1.4190155106891582) q[3];
ry(-1.9178930098056544) q[4];
cx q[3],q[4];
ry(1.6373249659948081) q[4];
ry(1.8270983282482653) q[5];
cx q[4],q[5];
ry(-1.0639646480990943) q[4];
ry(-1.2089671676541425) q[5];
cx q[4],q[5];
ry(1.5228670235879012) q[5];
ry(-0.504040865400019) q[6];
cx q[5],q[6];
ry(-1.6064856165916463) q[5];
ry(-0.8778799867738059) q[6];
cx q[5],q[6];
ry(-0.5383939871755716) q[6];
ry(-1.4605901647828823) q[7];
cx q[6],q[7];
ry(1.5760369257510012) q[6];
ry(-0.00019847919674464973) q[7];
cx q[6],q[7];
ry(3.11761548399863) q[7];
ry(1.574907472851784) q[8];
cx q[7],q[8];
ry(0.6194625378834866) q[7];
ry(-0.08182109429132023) q[8];
cx q[7],q[8];
ry(2.3602061842936237) q[8];
ry(1.5739381797455056) q[9];
cx q[8],q[9];
ry(1.4762679378116266) q[8];
ry(-3.136144708610354) q[9];
cx q[8],q[9];
ry(-0.10862510116222757) q[9];
ry(0.1517909541443913) q[10];
cx q[9],q[10];
ry(-1.3151843618039436) q[9];
ry(-3.141387892111299) q[10];
cx q[9],q[10];
ry(-2.401527128188484) q[10];
ry(-2.146865181086758) q[11];
cx q[10],q[11];
ry(-0.46506116820685184) q[10];
ry(3.1303220872524897) q[11];
cx q[10],q[11];
ry(0.7430387921836097) q[0];
ry(-2.397543775779161) q[1];
cx q[0],q[1];
ry(0.156730194437376) q[0];
ry(-0.6430573505370428) q[1];
cx q[0],q[1];
ry(-2.8346605351984624) q[1];
ry(-2.564970717642409) q[2];
cx q[1],q[2];
ry(0.314313951752428) q[1];
ry(-0.8143394573180087) q[2];
cx q[1],q[2];
ry(2.9347108349679707) q[2];
ry(1.6049018290568162) q[3];
cx q[2],q[3];
ry(-2.5150586315986097) q[2];
ry(-0.999897713147565) q[3];
cx q[2],q[3];
ry(-1.5415633354567586) q[3];
ry(2.805448518290455) q[4];
cx q[3],q[4];
ry(-3.1281883237951416) q[3];
ry(2.1145132013544954) q[4];
cx q[3],q[4];
ry(-0.2676532724625478) q[4];
ry(-1.0902546344560513) q[5];
cx q[4],q[5];
ry(3.139523800295689) q[4];
ry(0.14314313907296422) q[5];
cx q[4],q[5];
ry(1.9645459032252182) q[5];
ry(2.5424743525580875) q[6];
cx q[5],q[6];
ry(-0.0035353475138384383) q[5];
ry(2.436031394227493) q[6];
cx q[5],q[6];
ry(-1.7788540250974068) q[6];
ry(-1.3215111522851821) q[7];
cx q[6],q[7];
ry(-1.7033159967495215) q[6];
ry(2.5251349485079495) q[7];
cx q[6],q[7];
ry(1.6058566440221265) q[7];
ry(-0.7765890829731169) q[8];
cx q[7],q[8];
ry(1.56707962031739) q[7];
ry(-1.570740314320183) q[8];
cx q[7],q[8];
ry(1.6094169416673239) q[8];
ry(-2.1140192257382333) q[9];
cx q[8],q[9];
ry(-0.02868299881570202) q[8];
ry(-1.5617417300967125) q[9];
cx q[8],q[9];
ry(-2.7009264485403004) q[9];
ry(-1.2418311644051085) q[10];
cx q[9],q[10];
ry(-1.8250834683854666) q[9];
ry(1.6068477936795555) q[10];
cx q[9],q[10];
ry(0.042179318875337296) q[10];
ry(-1.2059818452557556) q[11];
cx q[10],q[11];
ry(2.729872851682051) q[10];
ry(2.621835371770506) q[11];
cx q[10],q[11];
ry(1.9944506342919288) q[0];
ry(1.4806354240878081) q[1];
cx q[0],q[1];
ry(-1.411439148409802) q[0];
ry(-2.159780003542496) q[1];
cx q[0],q[1];
ry(-1.1087749451939692) q[1];
ry(1.1640090014005122) q[2];
cx q[1],q[2];
ry(0.05839523529299091) q[1];
ry(0.02377329577983868) q[2];
cx q[1],q[2];
ry(-0.600599870648467) q[2];
ry(1.5792483065455638) q[3];
cx q[2],q[3];
ry(2.189274627954214) q[2];
ry(0.014889798777364405) q[3];
cx q[2],q[3];
ry(-2.841386987697606) q[3];
ry(1.561504865060016) q[4];
cx q[3],q[4];
ry(-1.580893625217029) q[3];
ry(-0.004083290026573927) q[4];
cx q[3],q[4];
ry(1.4416051287848273) q[4];
ry(1.4025118058067259) q[5];
cx q[4],q[5];
ry(-1.5710862116138244) q[4];
ry(0.19173908833410813) q[5];
cx q[4],q[5];
ry(1.5712501652065) q[5];
ry(1.5705503242595635) q[6];
cx q[5],q[6];
ry(-1.565904891441594) q[5];
ry(1.5750191087034535) q[6];
cx q[5],q[6];
ry(-2.7857143643276916) q[6];
ry(-1.5709952920156087) q[7];
cx q[6],q[7];
ry(1.5668800123873803) q[6];
ry(-3.1411250583059585) q[7];
cx q[6],q[7];
ry(0.6639174539723358) q[7];
ry(1.570029606680115) q[8];
cx q[7],q[8];
ry(1.569653319853269) q[7];
ry(0.00037517998331537683) q[8];
cx q[7],q[8];
ry(1.5418535933796862) q[8];
ry(-1.3218017538181588) q[9];
cx q[8],q[9];
ry(-2.3061858147344254) q[8];
ry(3.1008313496473594) q[9];
cx q[8],q[9];
ry(-2.173922458766392) q[9];
ry(-1.0252460823529868) q[10];
cx q[9],q[10];
ry(-1.5510083537172177) q[9];
ry(-0.009558147227889259) q[10];
cx q[9],q[10];
ry(-1.5636007364783537) q[10];
ry(0.2965700482899827) q[11];
cx q[10],q[11];
ry(-1.6054421352855979) q[10];
ry(2.607243293008439) q[11];
cx q[10],q[11];
ry(1.0946003721174709) q[0];
ry(-1.2745671778268122) q[1];
cx q[0],q[1];
ry(1.3370944252419976) q[0];
ry(2.3369173955112834) q[1];
cx q[0],q[1];
ry(2.8110522309790773) q[1];
ry(-1.611408156298831) q[2];
cx q[1],q[2];
ry(-3.114585392188054) q[1];
ry(-0.004262319204965337) q[2];
cx q[1],q[2];
ry(2.117966200999649) q[2];
ry(2.8536511130274804) q[3];
cx q[2],q[3];
ry(-1.7439893396145951) q[2];
ry(0.7874887376959476) q[3];
cx q[2],q[3];
ry(-2.7412219157641267) q[3];
ry(1.7131018507167717) q[4];
cx q[3],q[4];
ry(-1.5678579592849733) q[3];
ry(-3.140474771534497) q[4];
cx q[3],q[4];
ry(-1.5718135461482907) q[4];
ry(1.5695253293092364) q[5];
cx q[4],q[5];
ry(-1.5714415456029553) q[4];
ry(1.5854796590257325) q[5];
cx q[4],q[5];
ry(1.5732849314777473) q[5];
ry(0.3555872782775103) q[6];
cx q[5],q[6];
ry(-1.567625129986848) q[5];
ry(0.29261487994433416) q[6];
cx q[5],q[6];
ry(-1.5710651674330827) q[6];
ry(2.476607669370897) q[7];
cx q[6],q[7];
ry(-2.524127923695671) q[6];
ry(-0.6213209940654663) q[7];
cx q[6],q[7];
ry(-1.5714693165902351) q[7];
ry(1.5400313490717628) q[8];
cx q[7],q[8];
ry(1.5792670064529188) q[7];
ry(1.580043682200177) q[8];
cx q[7],q[8];
ry(-1.5839672263363884) q[8];
ry(-2.17454094955848) q[9];
cx q[8],q[9];
ry(3.141513833904769) q[8];
ry(-0.004533820620427929) q[9];
cx q[8],q[9];
ry(0.4109688664098457) q[9];
ry(-1.5648887159805747) q[10];
cx q[9],q[10];
ry(-0.7653195367990415) q[9];
ry(3.0938639644039774) q[10];
cx q[9],q[10];
ry(1.3682134482172517) q[10];
ry(-1.414917432714458) q[11];
cx q[10],q[11];
ry(-3.1017839918962675) q[10];
ry(3.0717769588452355) q[11];
cx q[10],q[11];
ry(3.008033614199955) q[0];
ry(0.40332476072983336) q[1];
ry(1.535797924310171) q[2];
ry(0.39741817567802684) q[3];
ry(-1.5701643259681193) q[4];
ry(1.56841658996) q[5];
ry(1.5709256305307475) q[6];
ry(1.5708592656886955) q[7];
ry(1.5830611381704103) q[8];
ry(1.6183402747720352) q[9];
ry(-1.7732696540871606) q[10];
ry(0.1523680232997) q[11];