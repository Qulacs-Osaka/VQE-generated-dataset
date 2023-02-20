OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.1942751431388727) q[0];
rz(1.3906589026279779) q[0];
ry(-0.5761467889377236) q[1];
rz(0.9482740263099776) q[1];
ry(0.00029363548747944704) q[2];
rz(1.7985016431458378) q[2];
ry(0.00033327598755938936) q[3];
rz(0.5284202437814948) q[3];
ry(1.5705061580526394) q[4];
rz(-0.735855153343867) q[4];
ry(1.571118336386341) q[5];
rz(-1.7718853400156103) q[5];
ry(-0.00012750172667974624) q[6];
rz(0.9997100849822314) q[6];
ry(3.141065649581013) q[7];
rz(-0.78375577039148) q[7];
ry(0.0006948657960759164) q[8];
rz(2.256383536121315) q[8];
ry(3.1413648445527413) q[9];
rz(-2.055422881664793) q[9];
ry(-3.1409885121545105) q[10];
rz(-0.8102505366145065) q[10];
ry(3.141090962800806) q[11];
rz(-1.9927311843375939) q[11];
ry(0.0396865856155868) q[12];
rz(-0.35105967458049075) q[12];
ry(-2.2037773991597156) q[13];
rz(0.9104247922164215) q[13];
ry(-2.9741796831526517) q[14];
rz(-0.6245848769073632) q[14];
ry(-2.662632820982427) q[15];
rz(2.925046658330139) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.4041323201324634) q[0];
rz(-2.9099648344583087) q[0];
ry(0.07035948739528752) q[1];
rz(-0.3256132015687017) q[1];
ry(-0.08765820766386945) q[2];
rz(2.2603085433866634) q[2];
ry(-2.668709716481108) q[3];
rz(-2.7291719032379613) q[3];
ry(-1.4352707786323462) q[4];
rz(-2.945493876295103) q[4];
ry(0.7455642084123522) q[5];
rz(-1.2251649244408336) q[5];
ry(-1.4215427185180645) q[6];
rz(-1.3048045057088637) q[6];
ry(1.7607723953160557) q[7];
rz(1.9664365172696716) q[7];
ry(-3.1414585967018516) q[8];
rz(-2.7489655430581) q[8];
ry(0.0006514371341266312) q[9];
rz(-2.825365318090728) q[9];
ry(-9.255484902765797e-05) q[10];
rz(-2.3164827698757153) q[10];
ry(-3.140979886909417) q[11];
rz(-2.0753744313323903) q[11];
ry(-1.2668836460853832) q[12];
rz(-1.6114910765712747) q[12];
ry(-3.1131749290036153) q[13];
rz(2.599118088433308) q[13];
ry(3.0934490123361784) q[14];
rz(1.3985385174136444) q[14];
ry(0.19549295650448328) q[15];
rz(0.6051555008169469) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.7988036969507295) q[0];
rz(-2.6095026479297743) q[0];
ry(-2.4678870931695887) q[1];
rz(-1.033542452754288) q[1];
ry(-0.11570301108335865) q[2];
rz(0.8945359915568273) q[2];
ry(0.6812946478429779) q[3];
rz(-2.4858928702039638) q[3];
ry(1.5356446049637074) q[4];
rz(2.271975810549799) q[4];
ry(1.6061835709804648) q[5];
rz(-2.275057514647462) q[5];
ry(1.9448181746883035) q[6];
rz(-2.034465456761943) q[6];
ry(-1.8366533499283655) q[7];
rz(-1.1403042549266071) q[7];
ry(3.1406811473558403) q[8];
rz(-0.6692256846176202) q[8];
ry(-3.1376271164582454) q[9];
rz(-1.5183974287097683) q[9];
ry(3.1413657363288405) q[10];
rz(-1.6399383869175157) q[10];
ry(-3.14126840566423) q[11];
rz(-0.017902759846711344) q[11];
ry(1.3093408655192809) q[12];
rz(-2.461682081114283) q[12];
ry(-1.4903669792377823) q[13];
rz(-0.2146512801673289) q[13];
ry(0.13754567941485432) q[14];
rz(-0.9151805085087314) q[14];
ry(-2.7454072849974462) q[15];
rz(-2.0807466665297745) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.4526812538424707) q[0];
rz(-3.1106998765658296) q[0];
ry(2.973203769690151) q[1];
rz(1.5409369922273852) q[1];
ry(-0.8744045597375791) q[2];
rz(-2.1784557229779047) q[2];
ry(0.988547406027267) q[3];
rz(0.27789645510194927) q[3];
ry(0.6170375274989309) q[4];
rz(0.18361422336001412) q[4];
ry(0.6182091485221148) q[5];
rz(2.967308519093762) q[5];
ry(0.21463544275966207) q[6];
rz(2.4416807925792727) q[6];
ry(3.10554700076566) q[7];
rz(-2.170120012512875) q[7];
ry(-1.5695829799406151) q[8];
rz(-3.0911170027878274) q[8];
ry(-1.569085429797025) q[9];
rz(-2.832580708983203) q[9];
ry(-3.141082853732879) q[10];
rz(-1.0960429106712528) q[10];
ry(3.1409129804830735) q[11];
rz(0.9738051212637978) q[11];
ry(-2.3795245198245585) q[12];
rz(1.4788953547562087) q[12];
ry(0.9325691260256012) q[13];
rz(-2.1739438370331303) q[13];
ry(1.5299159467883399) q[14];
rz(0.02597265021158712) q[14];
ry(0.9671490709818895) q[15];
rz(0.1687420302666615) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(2.452636041884302) q[0];
rz(0.17870034524759415) q[0];
ry(-1.163922481831601) q[1];
rz(-1.5712739892097642) q[1];
ry(1.597195468897155) q[2];
rz(-2.2068488929019816) q[2];
ry(0.20732039792133072) q[3];
rz(-0.8537963979770291) q[3];
ry(-2.8183220866577554) q[4];
rz(-1.0130200014053334) q[4];
ry(0.3198513120556808) q[5];
rz(2.3523312989008254) q[5];
ry(-1.8648416803851458) q[6];
rz(1.2005832288221217) q[6];
ry(1.4318507329995764) q[7];
rz(0.7598543154017054) q[7];
ry(-1.485379152614759) q[8];
rz(-1.633997390319533) q[8];
ry(-0.18322154581941574) q[9];
rz(-1.8162803761915085) q[9];
ry(-1.5481116572268556) q[10];
rz(-0.6358967504438741) q[10];
ry(1.5746657390860896) q[11];
rz(0.6360703392299372) q[11];
ry(0.7305702376104471) q[12];
rz(-1.8305356115563454) q[12];
ry(-2.3005106963094994) q[13];
rz(2.0816182195825856) q[13];
ry(-1.325869599558528) q[14];
rz(-3.040148372197773) q[14];
ry(2.9205865065907597) q[15];
rz(2.1796461505974665) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.1068860982415645) q[0];
rz(-0.5050762474102776) q[0];
ry(-1.5228509455322552) q[1];
rz(1.0607616130676352) q[1];
ry(2.1782500414984343) q[2];
rz(1.4523968469064634) q[2];
ry(1.519371218097559) q[3];
rz(-1.5263921649946015) q[3];
ry(0.013701831207177051) q[4];
rz(-2.1688877751440163) q[4];
ry(-0.010978224328513836) q[5];
rz(-2.3965726983988014) q[5];
ry(3.0276141514130623) q[6];
rz(0.6194178442298978) q[6];
ry(-2.930157932508023) q[7];
rz(-1.6401203610020247) q[7];
ry(2.502971006848077) q[8];
rz(1.8687765018090357) q[8];
ry(0.6482981778442714) q[9];
rz(2.0188053038983815) q[9];
ry(3.1327272629389964) q[10];
rz(1.7470651815557903) q[10];
ry(0.002989851457405513) q[11];
rz(-2.443985958385129) q[11];
ry(1.6815934822849399) q[12];
rz(-1.6836563110222276) q[12];
ry(-0.005293516161620569) q[13];
rz(1.1245727403261179) q[13];
ry(-0.822793816793489) q[14];
rz(-1.0101855033863423) q[14];
ry(-1.012389504885946) q[15];
rz(2.5509625038981456) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-0.6636720906369032) q[0];
rz(1.5491211679892753) q[0];
ry(-2.889626943386733) q[1];
rz(-0.9449532330897172) q[1];
ry(-2.2800970718808737) q[2];
rz(0.7370852010346791) q[2];
ry(1.7516375453642938) q[3];
rz(-1.4267243493828206) q[3];
ry(1.8278303240860367) q[4];
rz(2.5279154806123922) q[4];
ry(1.3137675969244835) q[5];
rz(0.6113129978208924) q[5];
ry(-2.0096877789549934) q[6];
rz(-2.2444958598795703) q[6];
ry(-1.1260324466365057) q[7];
rz(0.6081277449908895) q[7];
ry(2.9005086485153138) q[8];
rz(1.8364450246359374) q[8];
ry(-2.969104187340133) q[9];
rz(-1.1499838384406529) q[9];
ry(-0.31856916869339713) q[10];
rz(-0.2514035509462281) q[10];
ry(-1.8849771147940881) q[11];
rz(2.6961807855008435) q[11];
ry(-2.3843191463953564) q[12];
rz(-3.1368158311338012) q[12];
ry(2.0620758713825174) q[13];
rz(1.6602207593274203) q[13];
ry(1.7835820535541795) q[14];
rz(1.21704188071687) q[14];
ry(-0.9765364573776601) q[15];
rz(-2.8713062166094163) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-0.7971431130457753) q[0];
rz(2.421606906222675) q[0];
ry(-0.9533335657855088) q[1];
rz(0.8030066565424345) q[1];
ry(3.1103577299539253) q[2];
rz(1.8939808859920904) q[2];
ry(2.253961473168192) q[3];
rz(-2.905125032685112) q[3];
ry(-0.9258757543945414) q[4];
rz(-2.075977161851023) q[4];
ry(-2.2119365784077685) q[5];
rz(1.3251676827152488) q[5];
ry(1.5791127260325197) q[6];
rz(-2.919388476311366) q[6];
ry(-1.74008335125114) q[7];
rz(-0.21295458625142577) q[7];
ry(-2.489809152698011) q[8];
rz(2.6080678022056647) q[8];
ry(-2.6177446911716116) q[9];
rz(-0.6557273876148909) q[9];
ry(-3.124927472347013) q[10];
rz(1.618104017012415) q[10];
ry(-3.1248828776717845) q[11];
rz(1.6398190488358457) q[11];
ry(-0.030713888069730192) q[12];
rz(0.12438950800113803) q[12];
ry(3.140258991049977) q[13];
rz(-1.6347947538153411) q[13];
ry(1.5404458627700024) q[14];
rz(-0.07313223899812725) q[14];
ry(2.1249832633714805) q[15];
rz(1.5076248950996372) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-3.137324295454496) q[0];
rz(-2.8981381816773153) q[0];
ry(0.2910675000669762) q[1];
rz(-2.0010353833109846) q[1];
ry(-2.217917755102845) q[2];
rz(0.9484221328857786) q[2];
ry(-1.0078344313452978) q[3];
rz(-1.8670533940745013) q[3];
ry(-1.4405191768272996) q[4];
rz(2.955576221854161) q[4];
ry(1.5474663187867117) q[5];
rz(2.7703932605603505) q[5];
ry(0.18039196730764917) q[6];
rz(-2.4526732847977533) q[6];
ry(0.09659729618591939) q[7];
rz(2.0639918903476566) q[7];
ry(-0.08108869599445825) q[8];
rz(-1.2937515019904176) q[8];
ry(-3.043143535988864) q[9];
rz(-0.6690564533391156) q[9];
ry(-1.3517520831747705) q[10];
rz(-1.3909001947945736) q[10];
ry(-1.3130473655074342) q[11];
rz(-0.4843555260294928) q[11];
ry(-0.40889092890556206) q[12];
rz(-2.075374745815114) q[12];
ry(0.9692786635611296) q[13];
rz(-1.136813808760011) q[13];
ry(2.5010154837202894) q[14];
rz(2.014910826528946) q[14];
ry(-2.871068190814125) q[15];
rz(0.3776035605343173) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.6379117252961615) q[0];
rz(-1.6415581364240974) q[0];
ry(2.804884717975856) q[1];
rz(-0.956067817588582) q[1];
ry(3.136957864544592) q[2];
rz(-2.375631064620141) q[2];
ry(0.005645319274769094) q[3];
rz(-0.942318557002076) q[3];
ry(-0.0021288473796987617) q[4];
rz(1.3264341753017361) q[4];
ry(0.004840390785331011) q[5];
rz(-1.4962996807182032) q[5];
ry(3.141315097114283) q[6];
rz(-1.9475845209879452) q[6];
ry(-0.005945758727072814) q[7];
rz(1.4758955730740264) q[7];
ry(0.01732617498066169) q[8];
rz(-2.0205787643216113) q[8];
ry(2.8649674343344107) q[9];
rz(2.2661258284438257) q[9];
ry(3.050455944159396) q[10];
rz(-0.9905325391765957) q[10];
ry(3.0683816554212173) q[11];
rz(-0.7898693485042055) q[11];
ry(-2.4093294427961105) q[12];
rz(1.0393247930206346) q[12];
ry(0.8991583065278066) q[13];
rz(3.109252771571018) q[13];
ry(-1.3557053686626024) q[14];
rz(2.325746344622992) q[14];
ry(-1.3717505175113929) q[15];
rz(-1.586458752379289) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.6022383312192687) q[0];
rz(-2.1930639070813656) q[0];
ry(-1.3893164847426773) q[1];
rz(-2.324942149295322) q[1];
ry(2.5639490882605447) q[2];
rz(-0.4704789840907856) q[2];
ry(0.9936226866219329) q[3];
rz(1.7922396990354343) q[3];
ry(-2.1692955432155756) q[4];
rz(-1.5896782138733734) q[4];
ry(-2.446276589735249) q[5];
rz(-1.3157058453978452) q[5];
ry(2.3588526605894664) q[6];
rz(0.1396503251223903) q[6];
ry(-0.8482192094362324) q[7];
rz(-0.409019761617642) q[7];
ry(2.9451260177961727) q[8];
rz(-0.14414018626838063) q[8];
ry(2.9118771322859627) q[9];
rz(-0.6660410119036612) q[9];
ry(3.1355787089970746) q[10];
rz(-0.9410264971207535) q[10];
ry(0.012111505209176698) q[11];
rz(-0.26288686346816625) q[11];
ry(-3.11807746132116) q[12];
rz(1.1549987957547325) q[12];
ry(-3.1380796191945426) q[13];
rz(-0.7736903993304974) q[13];
ry(2.8695013656302075) q[14];
rz(-0.32222201835719655) q[14];
ry(1.9732336034636235) q[15];
rz(2.327779203186282) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.939742627353855) q[0];
rz(1.800852741541194) q[0];
ry(-0.506556345397153) q[1];
rz(-1.944547637314635) q[1];
ry(-0.05433642843996456) q[2];
rz(-0.6468951070079073) q[2];
ry(3.1413115270498753) q[3];
rz(0.18372674348336826) q[3];
ry(-3.126888388622848) q[4];
rz(-2.22716350643987) q[4];
ry(-3.1289090795142065) q[5];
rz(-2.433194208258418) q[5];
ry(-3.138593205533028) q[6];
rz(-1.090134261583168) q[6];
ry(-0.005482123717651849) q[7];
rz(-2.7860516737620187) q[7];
ry(0.1355258391938543) q[8];
rz(-0.3772468448071802) q[8];
ry(-2.7020037812149917) q[9];
rz(-0.5611527476108407) q[9];
ry(3.115272773220914) q[10];
rz(0.4431720478209469) q[10];
ry(-0.08024745269811273) q[11];
rz(-0.009043215734331868) q[11];
ry(-2.705048018563824) q[12];
rz(-0.2671048056290646) q[12];
ry(-0.18389307315120976) q[13];
rz(1.7696832937794111) q[13];
ry(1.6602616958974616) q[14];
rz(-0.7192536774951609) q[14];
ry(1.8966931471181478) q[15];
rz(0.870584326972269) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.9637791126291315) q[0];
rz(-2.192363249909903) q[0];
ry(0.5173761227084475) q[1];
rz(-0.3028528856867441) q[1];
ry(0.05645169186989549) q[2];
rz(1.0314578530436878) q[2];
ry(2.7053629936685857) q[3];
rz(1.575797208886299) q[3];
ry(0.08431418950122678) q[4];
rz(-2.2802530048869585) q[4];
ry(-2.8661774067368753) q[5];
rz(0.18085399557773146) q[5];
ry(-2.8524618875599823) q[6];
rz(1.3870796593452699) q[6];
ry(0.7274493027670466) q[7];
rz(-2.897289632224851) q[7];
ry(2.920992973957756) q[8];
rz(2.9399626278105915) q[8];
ry(0.31818737243391393) q[9];
rz(0.0444199119214629) q[9];
ry(-2.2750894301009073) q[10];
rz(-1.8161812600038285) q[10];
ry(1.9988726748046286) q[11];
rz(0.7957879160673542) q[11];
ry(1.7209303869539188) q[12];
rz(0.9499795504960942) q[12];
ry(-2.275575482745838) q[13];
rz(-1.1152576687080813) q[13];
ry(2.5998966568358677) q[14];
rz(0.5336981825804803) q[14];
ry(-0.5048581212619565) q[15];
rz(1.0859223541694254) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(2.8382119016529272) q[0];
rz(-0.5757899711277384) q[0];
ry(-2.587672095238999) q[1];
rz(-0.5126647163696761) q[1];
ry(0.649766079307633) q[2];
rz(-2.303471503753125) q[2];
ry(2.482565595317384) q[3];
rz(2.325438765301727) q[3];
ry(0.00016916573612126514) q[4];
rz(1.5998100149251595) q[4];
ry(-3.1400298677179244) q[5];
rz(-0.9422805360001417) q[5];
ry(-0.008132295360193886) q[6];
rz(0.8710502697166722) q[6];
ry(-0.04010322938740706) q[7];
rz(3.0937105402093352) q[7];
ry(-3.1322167272162207) q[8];
rz(-1.7664097446075857) q[8];
ry(-3.1368368197921024) q[9];
rz(1.027862572581939) q[9];
ry(3.0593777511514535) q[10];
rz(2.8022016347483025) q[10];
ry(-3.076364874448291) q[11];
rz(-1.6438363449213191) q[11];
ry(-3.1408688122672164) q[12];
rz(-2.7300171169013483) q[12];
ry(3.127705193061771) q[13];
rz(-1.7102025895871762) q[13];
ry(2.9915064689395283) q[14];
rz(2.530627811154909) q[14];
ry(0.03501516112009906) q[15];
rz(-0.6558925576067335) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.796923940635791) q[0];
rz(3.1281534429529065) q[0];
ry(2.0426480515305956) q[1];
rz(-2.9849762829904773) q[1];
ry(-1.3380648176873033) q[2];
rz(2.323404527386079) q[2];
ry(0.5264203991928627) q[3];
rz(0.24220906836244982) q[3];
ry(1.66933680023414) q[4];
rz(-1.529057197145393) q[4];
ry(-3.080662027410492) q[5];
rz(-1.8678471824515226) q[5];
ry(2.646897993045115) q[6];
rz(-1.8474519134199605) q[6];
ry(-2.185947808961526) q[7];
rz(-1.8746255543256822) q[7];
ry(-0.2139149658623447) q[8];
rz(-2.2536161897267206) q[8];
ry(0.18078340455307007) q[9];
rz(2.1484068065525106) q[9];
ry(2.116626010890176) q[10];
rz(2.532701082483001) q[10];
ry(-3.112138597424085) q[11];
rz(-1.102939857761231) q[11];
ry(1.4148191875627105) q[12];
rz(-2.155779114469344) q[12];
ry(-0.7281323673109696) q[13];
rz(0.7714968107857787) q[13];
ry(1.6680432032785846) q[14];
rz(-1.4395618023931345) q[14];
ry(2.9676357035020757) q[15];
rz(-0.6568610330962358) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-2.7763765360703077) q[0];
rz(-0.227558170435558) q[0];
ry(2.7346101381385215) q[1];
rz(2.338859898253571) q[1];
ry(0.16578064859650077) q[2];
rz(2.0906414093286663) q[2];
ry(-2.9749746558134067) q[3];
rz(0.03207957888828705) q[3];
ry(0.0015788153815646968) q[4];
rz(0.8146808303518379) q[4];
ry(-3.135553226923727) q[5];
rz(0.049291800957515945) q[5];
ry(-0.8013789402813405) q[6];
rz(-1.349470536330166) q[6];
ry(0.2970208350658421) q[7];
rz(2.07142010293262) q[7];
ry(1.5261233050228076) q[8];
rz(2.697300914924753) q[8];
ry(1.5108817758888222) q[9];
rz(0.44648385840792754) q[9];
ry(3.1374144899990237) q[10];
rz(-2.649351198543685) q[10];
ry(0.2315025485113736) q[11];
rz(1.8870646126865671) q[11];
ry(2.722410645114381) q[12];
rz(-0.5340434267435611) q[12];
ry(2.709395860532813) q[13];
rz(0.9976476638868877) q[13];
ry(-1.6737794766097573) q[14];
rz(-1.8562389769270535) q[14];
ry(-3.093443790352214) q[15];
rz(0.6544776888636662) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-2.839093079030178) q[0];
rz(1.1177801500226074) q[0];
ry(-0.9568715590760828) q[1];
rz(2.028418571899179) q[1];
ry(0.7362697335923567) q[2];
rz(-2.015801181645688) q[2];
ry(2.925904971217195) q[3];
rz(-2.855440317950324) q[3];
ry(3.104703626104533) q[4];
rz(2.9058002343825113) q[4];
ry(-3.044886558950836) q[5];
rz(1.7774976491325967) q[5];
ry(0.0038551394557776795) q[6];
rz(-3.120988020645946) q[6];
ry(0.00021263614900748196) q[7];
rz(-1.5177022429074203) q[7];
ry(-3.139645225624314) q[8];
rz(-3.132723599190721) q[8];
ry(-0.0014223813737462726) q[9];
rz(-1.9882329634730638) q[9];
ry(-0.011141730192601408) q[10];
rz(-2.7707165943911796) q[10];
ry(0.01183387059425911) q[11];
rz(2.2050906577748606) q[11];
ry(-2.5996345811804265) q[12];
rz(0.1469663174006819) q[12];
ry(2.81176504395616) q[13];
rz(-1.5266024567548142) q[13];
ry(0.25312280073476057) q[14];
rz(-0.9366414524336498) q[14];
ry(1.2733814616623302) q[15];
rz(1.9070071378870623) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-0.21064084117314172) q[0];
rz(-0.8078425168502251) q[0];
ry(1.3273148666875052) q[1];
rz(-0.40702420476828055) q[1];
ry(-1.73047457324924) q[2];
rz(-0.8528163457361622) q[2];
ry(1.4092509293040192) q[3];
rz(-1.7483233384944268) q[3];
ry(3.120152116502745) q[4];
rz(1.5841892845359207) q[4];
ry(3.1027554157000057) q[5];
rz(3.060742700773694) q[5];
ry(-2.728093184037004) q[6];
rz(2.540731330035313) q[6];
ry(-0.5600503003979576) q[7];
rz(1.1373845106805398) q[7];
ry(2.6711739373732937) q[8];
rz(2.432036975267458) q[8];
ry(-0.7046334523566307) q[9];
rz(-2.047213692818627) q[9];
ry(-3.1391425366580106) q[10];
rz(-2.4235948225696555) q[10];
ry(0.01203618165370276) q[11];
rz(-2.023413176183056) q[11];
ry(0.029026772131144796) q[12];
rz(1.5730246356546334) q[12];
ry(3.1009243599712213) q[13];
rz(2.3333342175197163) q[13];
ry(3.0056278860132353) q[14];
rz(-2.4230551122836363) q[14];
ry(-3.1020658420120286) q[15];
rz(0.9916410548448633) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(3.12726433157667) q[0];
rz(-1.9068187633956526) q[0];
ry(1.7916915045329302) q[1];
rz(-0.7593650891556655) q[1];
ry(3.119350620221538) q[2];
rz(-2.5488130185364253) q[2];
ry(3.110152815631813) q[3];
rz(-2.728253518579426) q[3];
ry(-2.1472568702702612) q[4];
rz(2.5680970381383994) q[4];
ry(-1.8883941134051234) q[5];
rz(0.030929415314116217) q[5];
ry(-3.0735996804674413) q[6];
rz(0.7421479544988508) q[6];
ry(-0.018011994600428242) q[7];
rz(1.898498217614181) q[7];
ry(-0.06099400930976277) q[8];
rz(-2.9609462988406134) q[8];
ry(-0.0067937862476504405) q[9];
rz(2.7585674616093963) q[9];
ry(2.983368320492838) q[10];
rz(-2.819346310142217) q[10];
ry(-0.16816559281715654) q[11];
rz(3.001284564448667) q[11];
ry(-2.4330747351265902) q[12];
rz(-1.9328982937530697) q[12];
ry(-0.36559842204699233) q[13];
rz(0.7276750448068522) q[13];
ry(-0.4621906278941701) q[14];
rz(2.8241977565538146) q[14];
ry(-0.9662960408033899) q[15];
rz(-1.099558333446053) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.4421759187050074) q[0];
rz(-1.7705457691737194) q[0];
ry(-3.1286441065878026) q[1];
rz(0.6974373605032174) q[1];
ry(-1.1144286017734562) q[2];
rz(1.9332347559587115) q[2];
ry(1.5964219606296757) q[3];
rz(-2.421414452187155) q[3];
ry(-0.03322112755266524) q[4];
rz(-0.8750625344388201) q[4];
ry(-0.027962027912875498) q[5];
rz(-0.040187803539428295) q[5];
ry(2.9531539071512056) q[6];
rz(0.7198257772433101) q[6];
ry(0.2809038663478951) q[7];
rz(-0.5518147248732248) q[7];
ry(-1.0720984966888174) q[8];
rz(-0.8757402710376904) q[8];
ry(-1.837264879888357) q[9];
rz(1.5506218697115375) q[9];
ry(1.531800249585883) q[10];
rz(2.9305093597232177) q[10];
ry(1.6507673440168258) q[11];
rz(2.8119073476729963) q[11];
ry(-2.950637463798207) q[12];
rz(-2.731026484370023) q[12];
ry(-0.6950609220068058) q[13];
rz(-1.7726339871413808) q[13];
ry(-0.39498320093668937) q[14];
rz(1.067323659975119) q[14];
ry(-0.32995532862248256) q[15];
rz(-1.0480314779318354) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-2.020048387751057) q[0];
rz(-1.5263104247478347) q[0];
ry(1.1121006695728353) q[1];
rz(-0.8725702339876837) q[1];
ry(-1.4000436789843376) q[2];
rz(-0.007439778684478426) q[2];
ry(-2.5442691382985396) q[3];
rz(0.8923579222689888) q[3];
ry(-3.11079727712458) q[4];
rz(0.9002836058489718) q[4];
ry(-0.032588458999750894) q[5];
rz(-1.6500032151401596) q[5];
ry(0.0632815863758669) q[6];
rz(-2.5147569924409807) q[6];
ry(-0.01119150109371753) q[7];
rz(2.147334617157594) q[7];
ry(-1.574513035105589) q[8];
rz(-3.0233470910596187) q[8];
ry(-1.548927071624508) q[9];
rz(2.9579604589548727) q[9];
ry(-0.011094141558360526) q[10];
rz(-3.136225204453596) q[10];
ry(-0.009948290569668276) q[11];
rz(2.8747571224801476) q[11];
ry(-2.046599452375218) q[12];
rz(1.5063548890811918) q[12];
ry(0.970424792185015) q[13];
rz(-3.1210476409031633) q[13];
ry(3.0980799170306095) q[14];
rz(-2.1658511804527762) q[14];
ry(-0.0032054296769388206) q[15];
rz(-1.9954169331709188) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-3.0990179844179475) q[0];
rz(-0.5087615195198126) q[0];
ry(-3.073190682309742) q[1];
rz(-1.297043567549573) q[1];
ry(-1.6357101870057296) q[2];
rz(-1.8792837132114115) q[2];
ry(-1.56681169142895) q[3];
rz(-2.395778935050584) q[3];
ry(3.079157101980439) q[4];
rz(-3.022424544717458) q[4];
ry(3.1315599288922398) q[5];
rz(-2.2263753113113065) q[5];
ry(0.9416169260306395) q[6];
rz(-2.987239055962125) q[6];
ry(-2.4209347315605014) q[7];
rz(-0.31656709802156385) q[7];
ry(1.4932905217813082) q[8];
rz(2.8775669352317474) q[8];
ry(-1.6269723877082656) q[9];
rz(-0.4088542348419829) q[9];
ry(3.133523369244117) q[10];
rz(-0.13643093058554595) q[10];
ry(3.013760954606488) q[11];
rz(-1.1678806079648534) q[11];
ry(2.2155653963812174) q[12];
rz(-1.608821171865125) q[12];
ry(2.8763579573229747) q[13];
rz(-0.44696936376091495) q[13];
ry(-2.9963941745898435) q[14];
rz(-2.7859615818521983) q[14];
ry(0.04829923855052014) q[15];
rz(1.3491440282821223) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-3.141314412960808) q[0];
rz(2.8340629860575803) q[0];
ry(3.140152109917375) q[1];
rz(-2.397772440817341) q[1];
ry(-1.3229259790444208) q[2];
rz(0.3558534872108991) q[2];
ry(1.5820595865391978) q[3];
rz(0.31353241995140846) q[3];
ry(-0.0002485823818242054) q[4];
rz(-1.0881386121494527) q[4];
ry(-0.0003147457129625408) q[5];
rz(-1.373108851643197) q[5];
ry(-3.0754368144424524) q[6];
rz(2.0555779441247757) q[6];
ry(-3.086298806904974) q[7];
rz(-0.8366873455421073) q[7];
ry(-0.0702238274973146) q[8];
rz(-2.7570955783563873) q[8];
ry(1.3739257683787869) q[9];
rz(-1.2437133135669105) q[9];
ry(-1.6181936995783497) q[10];
rz(-2.5128745609080267) q[10];
ry(1.5209300980339477) q[11];
rz(1.816417770323568) q[11];
ry(-1.4445508240260434) q[12];
rz(-1.1718464550249026) q[12];
ry(-1.4988380515096438) q[13];
rz(2.524060731605197) q[13];
ry(2.67019684992157) q[14];
rz(-2.888967613633299) q[14];
ry(2.2543826504376954) q[15];
rz(-2.0461268275263995) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(2.660068680279129) q[0];
rz(1.7132266078815754) q[0];
ry(0.2890861808615046) q[1];
rz(-1.3145848945958047) q[1];
ry(-1.1868500149094166) q[2];
rz(1.6970168456127386) q[2];
ry(1.316331144934595) q[3];
rz(-1.8139434317603802) q[3];
ry(-0.10074477335582267) q[4];
rz(1.2609002785334091) q[4];
ry(-0.25781259625403674) q[5];
rz(2.5870124372647054) q[5];
ry(0.00015490264646094668) q[6];
rz(-1.2467777657614976) q[6];
ry(3.092518884897813) q[7];
rz(1.6609709445740186) q[7];
ry(0.06057566914594302) q[8];
rz(1.7208130686923813) q[8];
ry(-3.1317036526174995) q[9];
rz(-2.804976385369767) q[9];
ry(-0.01244721264562628) q[10];
rz(-0.6538242898541454) q[10];
ry(0.05359166162577935) q[11];
rz(1.4394549144233473) q[11];
ry(-2.454762880193428) q[12];
rz(2.707375839521912) q[12];
ry(-0.7391462146505872) q[13];
rz(0.3855936273402843) q[13];
ry(1.5457465050494246) q[14];
rz(-0.5866700487215326) q[14];
ry(-1.679724033432799) q[15];
rz(0.37832568564476904) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(2.6313317940888714) q[0];
rz(2.620000274157613) q[0];
ry(2.6309497660887207) q[1];
rz(-0.8134421332862553) q[1];
ry(-0.015345610375463881) q[2];
rz(-0.778020866277491) q[2];
ry(3.124409834266951) q[3];
rz(1.8707224358123051) q[3];
ry(-0.02777916662455754) q[4];
rz(1.1886169065835963) q[4];
ry(3.115901147928585) q[5];
rz(-2.5369458720196696) q[5];
ry(3.1398226213074993) q[6];
rz(2.035138137576987) q[6];
ry(0.006859791676084193) q[7];
rz(2.4713890335374353) q[7];
ry(-1.5394155193279746) q[8];
rz(-0.24915780012010677) q[8];
ry(1.5943473838818174) q[9];
rz(-1.1722026361283262) q[9];
ry(1.6509570868684627) q[10];
rz(0.7243871863525482) q[10];
ry(1.6053430207366868) q[11];
rz(-2.9547166745658235) q[11];
ry(3.0651247255175846) q[12];
rz(0.18750472339330848) q[12];
ry(-3.0451518893038534) q[13];
rz(-0.24194187288934008) q[13];
ry(-1.5635289430021135) q[14];
rz(-0.6936298194119975) q[14];
ry(1.6429571135018302) q[15];
rz(2.522602884115984) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(2.6444512659832604) q[0];
rz(3.0565363921395265) q[0];
ry(-2.538931504055656) q[1];
rz(-0.4195154153464422) q[1];
ry(2.3755510549178167) q[2];
rz(-0.30973895036175847) q[2];
ry(0.2813832378011991) q[3];
rz(-2.82427900856109) q[3];
ry(2.9456851708344183) q[4];
rz(1.8152083581266494) q[4];
ry(-2.9169902958227305) q[5];
rz(-2.311920579063135) q[5];
ry(2.2312084431359365) q[6];
rz(2.4821590000213924) q[6];
ry(-0.8863533282724374) q[7];
rz(2.714698741294057) q[7];
ry(-0.925595259271964) q[8];
rz(-1.09284965922485) q[8];
ry(-2.240460279900843) q[9];
rz(1.8753398771848628) q[9];
ry(-1.8937345492291753) q[10];
rz(0.03383191984184997) q[10];
ry(1.2510367644756817) q[11];
rz(-0.012074726697016312) q[11];
ry(0.779449186348107) q[12];
rz(-2.1294435285082907) q[12];
ry(-2.361221129156104) q[13];
rz(-2.087384093044723) q[13];
ry(1.3921831289257804) q[14];
rz(1.7666368699846347) q[14];
ry(-1.2743423063487767) q[15];
rz(-1.497784421055436) q[15];