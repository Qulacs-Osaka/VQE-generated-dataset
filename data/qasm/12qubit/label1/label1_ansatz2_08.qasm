OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.005599551587335329) q[0];
rz(2.4177652271963264) q[0];
ry(-0.008584719697083187) q[1];
rz(0.18390930389936155) q[1];
ry(3.136012323088457) q[2];
rz(-0.13280176202134153) q[2];
ry(3.133993898138549) q[3];
rz(0.661048724254078) q[3];
ry(-0.060054622131466666) q[4];
rz(-1.6165718941105673) q[4];
ry(-0.0027418177377738218) q[5];
rz(-0.37629905702891414) q[5];
ry(-3.0456881656034196) q[6];
rz(-1.4513529127807718) q[6];
ry(1.6335957128314935) q[7];
rz(2.578155863011038) q[7];
ry(-1.0411613068843448) q[8];
rz(-1.9639956369247473) q[8];
ry(0.0138065038528209) q[9];
rz(0.4184302570268916) q[9];
ry(0.323987127579378) q[10];
rz(-2.9996097209559043) q[10];
ry(0.21756016034700967) q[11];
rz(3.0188533271734617) q[11];
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
ry(-3.132864765817155) q[0];
rz(-2.613806989709837) q[0];
ry(3.1227033946724014) q[1];
rz(0.20661779460080873) q[1];
ry(-3.128105462854311) q[2];
rz(2.8238322370453504) q[2];
ry(0.004494081350749468) q[3];
rz(-0.5018852854840069) q[3];
ry(-0.018655900708949562) q[4];
rz(-2.53449408807805) q[4];
ry(-0.0024303697395052296) q[5];
rz(2.108391857409134) q[5];
ry(-3.1053406862535726) q[6];
rz(-2.4913451156435764) q[6];
ry(-1.7783175829070683) q[7];
rz(1.2927032808752115) q[7];
ry(-0.41455346676397387) q[8];
rz(0.6061949599185131) q[8];
ry(-0.009661820776296182) q[9];
rz(-2.4208007755936736) q[9];
ry(1.1258652641333287) q[10];
rz(1.1706753094663185) q[10];
ry(2.1943722143599773) q[11];
rz(2.5285176993881753) q[11];
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
ry(3.130034123694521) q[0];
rz(2.5694201169007895) q[0];
ry(-0.02374242922788028) q[1];
rz(0.6059661863349519) q[1];
ry(0.019919030429395693) q[2];
rz(-0.5876248404944355) q[2];
ry(-3.123486189742007) q[3];
rz(2.871739373051175) q[3];
ry(3.1325496021292243) q[4];
rz(0.30523249760785137) q[4];
ry(-3.141372782309899) q[5];
rz(1.1402670312249967) q[5];
ry(-3.1393064159395347) q[6];
rz(0.4838054503235476) q[6];
ry(2.4567561646356517) q[7];
rz(0.3095305609802832) q[7];
ry(2.541362224117026) q[8];
rz(0.24477225637521016) q[8];
ry(-1.5271067621322807) q[9];
rz(0.009398224452092098) q[9];
ry(1.5141853479357856) q[10];
rz(1.2498107503202407) q[10];
ry(-2.465483362191458) q[11];
rz(-0.024873405772225573) q[11];
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
ry(0.5574660943001959) q[0];
rz(-0.002165526611309865) q[0];
ry(1.063456629238675) q[1];
rz(-3.107993223131008) q[1];
ry(0.6715921182270925) q[2];
rz(0.046543798459929064) q[2];
ry(-0.2640069856707939) q[3];
rz(0.07166523258028056) q[3];
ry(-0.114963625419361) q[4];
rz(-3.1215476341057387) q[4];
ry(-0.00047438861652931866) q[5];
rz(-2.97926993181974) q[5];
ry(-0.05367508398652986) q[6];
rz(-2.427125211174775) q[6];
ry(0.013186986289179359) q[7];
rz(-0.4125145056585176) q[7];
ry(0.04432763209142365) q[8];
rz(2.427696938520618) q[8];
ry(-1.5470707576538674) q[9];
rz(-2.76496565284539) q[9];
ry(-0.03499654493948141) q[10];
rz(0.4903924837631246) q[10];
ry(3.107957026652882) q[11];
rz(2.3136113391190576) q[11];
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
ry(-2.8762292408461354) q[0];
rz(-1.5872834691450033) q[0];
ry(0.8035961763488788) q[1];
rz(-3.093012763425857) q[1];
ry(0.9651661380708755) q[2];
rz(0.0285788676695935) q[2];
ry(2.944186089301101) q[3];
rz(-3.13302447466294) q[3];
ry(0.08402965232073711) q[4];
rz(-1.5859172984159118) q[4];
ry(-0.0006176044266593874) q[5];
rz(2.9065972028864504) q[5];
ry(3.1317860472692245) q[6];
rz(0.6785636842373003) q[6];
ry(3.0929805588880397) q[7];
rz(2.6774400281260093) q[7];
ry(-3.118097273472402) q[8];
rz(-3.1393635072898447) q[8];
ry(-0.06081498763631199) q[9];
rz(-2.913109574573985) q[9];
ry(-3.132163605839582) q[10];
rz(-1.9006324739070148) q[10];
ry(0.025462225336930544) q[11];
rz(2.663782850705508) q[11];
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
ry(1.5742517182312419) q[0];
rz(1.4775434158360756) q[0];
ry(0.38821900985471736) q[1];
rz(-0.08473635775358272) q[1];
ry(-2.5029702675960643) q[2];
rz(-3.1085648562530936) q[2];
ry(1.9438256717606206) q[3];
rz(-3.09343565201105) q[3];
ry(1.567357847981371) q[4];
rz(2.9956124505145723) q[4];
ry(0.0008809558999591928) q[5];
rz(-2.344721070283998) q[5];
ry(-0.2513849917650779) q[6];
rz(0.12043165059104233) q[6];
ry(2.966556852472257) q[7];
rz(2.8192190935767067) q[7];
ry(3.046691627558726) q[8];
rz(0.5369647164941167) q[8];
ry(0.05331049749179757) q[9];
rz(-1.385733821580368) q[9];
ry(-3.117738028713396) q[10];
rz(2.3824693525657534) q[10];
ry(-3.1322214938676125) q[11];
rz(-2.5673554512829218) q[11];
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
ry(-2.464941871049471) q[0];
rz(1.2871436620476666) q[0];
ry(-0.19749191607470706) q[1];
rz(0.09301139558551431) q[1];
ry(-0.15553619732645296) q[2];
rz(-0.5695986810309268) q[2];
ry(-0.04091465474509369) q[3];
rz(3.1129496991739782) q[3];
ry(0.22457531069744796) q[4];
rz(3.040409353567155) q[4];
ry(0.0001677347949647774) q[5];
rz(-2.379935378754626) q[5];
ry(0.010379638814497662) q[6];
rz(-3.05948330157149) q[6];
ry(-3.127052391956438) q[7];
rz(-0.024496031135042468) q[7];
ry(-3.133093401437525) q[8];
rz(-0.008350118199755318) q[8];
ry(-3.1413957598163984) q[9];
rz(-2.2113495346612537) q[9];
ry(7.208723851141906e-06) q[10];
rz(1.7204729282818043) q[10];
ry(3.141544051551142) q[11];
rz(-1.4422484805063567) q[11];
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
ry(1.5685407603796793) q[0];
rz(2.1947477715758525) q[0];
ry(-3.058287280110236) q[1];
rz(-3.129790992294048) q[1];
ry(0.012439817323284252) q[2];
rz(0.09559780102571157) q[2];
ry(-1.0574121130749647) q[3];
rz(-3.08741569172535) q[3];
ry(1.5353618854538398) q[4];
rz(1.6464084786722202) q[4];
ry(0.004745520018021499) q[5];
rz(-0.1965762254743289) q[5];
ry(-3.0049750471687617) q[6];
rz(-3.0491949211563707) q[6];
ry(2.2065294642507167) q[7];
rz(-0.020424684676062732) q[7];
ry(0.15206158236842873) q[8];
rz(-2.747355839767548) q[8];
ry(0.041534552380938256) q[9];
rz(0.9343937956640619) q[9];
ry(-0.11022810690544915) q[10];
rz(1.1157313777575615) q[10];
ry(-0.18107761745711848) q[11];
rz(-0.024403567483938883) q[11];
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
ry(0.00683108089430549) q[0];
rz(-3.1035175128993333) q[0];
ry(0.04318149068397226) q[1];
rz(-2.7874646407252626) q[1];
ry(0.029848803398209608) q[2];
rz(0.46278992143872655) q[2];
ry(-0.24322891378159178) q[3];
rz(-3.121803883858516) q[3];
ry(-0.41635659303575157) q[4];
rz(0.03078197567464258) q[4];
ry(0.005221113470834204) q[5];
rz(-3.0888779648350377) q[5];
ry(-2.017089998759957) q[6];
rz(0.035013775575197796) q[6];
ry(0.3409944207975206) q[7];
rz(-3.0773622793433884) q[7];
ry(2.9628248361637572) q[8];
rz(-3.0615723869602656) q[8];
ry(-2.940005589206211) q[9];
rz(-2.8163796121091584) q[9];
ry(0.4170478029805735) q[10];
rz(1.7997670980504759) q[10];
ry(2.927376985775584) q[11];
rz(-0.3011315949820004) q[11];
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
ry(0.0011116761077394324) q[0];
rz(2.112963465582963) q[0];
ry(3.122874142789633) q[1];
rz(0.7551826588410497) q[1];
ry(0.009366098120485056) q[2];
rz(2.045059348246464) q[2];
ry(-0.4870984163310279) q[3];
rz(-0.1004943785738604) q[3];
ry(1.206826581375714) q[4];
rz(3.037962795089227) q[4];
ry(1.5764249527663656) q[5];
rz(-3.1411310913776957) q[5];
ry(1.9272248681759678) q[6];
rz(0.852843679344109) q[6];
ry(3.0513290709781558) q[7];
rz(-2.3623282732260122) q[7];
ry(3.1331966667421653) q[8];
rz(1.6764331434940853) q[8];
ry(-0.0034187127638780623) q[9];
rz(-2.0599419810992394) q[9];
ry(-3.1264652236299906) q[10];
rz(-3.035894809355917) q[10];
ry(-3.118033963008658) q[11];
rz(-0.11465960846477241) q[11];
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
ry(-3.1412638755559343) q[0];
rz(0.8508850084040879) q[0];
ry(-0.0029009508403432496) q[1];
rz(1.5095196551812657) q[1];
ry(-0.0029735659038170326) q[2];
rz(0.5931393728764957) q[2];
ry(-3.1347482446423993) q[3];
rz(-2.980255953573729) q[3];
ry(-3.113801769733085) q[4];
rz(0.005333916699517439) q[4];
ry(-1.6008775379947275) q[5];
rz(0.17335278532637055) q[5];
ry(0.02842899518819575) q[6];
rz(-0.19124082355076666) q[6];
ry(-3.1137583315105526) q[7];
rz(0.21723608358066748) q[7];
ry(0.09158368878792882) q[8];
rz(0.06561983724061773) q[8];
ry(-0.1895045042955008) q[9];
rz(3.106617175761804) q[9];
ry(-0.4855154891272276) q[10];
rz(3.1404681093992646) q[10];
ry(0.9670928969835488) q[11];
rz(-0.0060048200884379455) q[11];
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
ry(-3.1415705219540144) q[0];
rz(0.3004676595107776) q[0];
ry(-0.004579332888591914) q[1];
rz(-2.9062929611534485) q[1];
ry(-0.006758154713270237) q[2];
rz(2.737826865850518) q[2];
ry(-0.01659668173369866) q[3];
rz(2.0668536343431096) q[3];
ry(0.051556243284596626) q[4];
rz(-0.9143166615673352) q[4];
ry(-0.046675296172242954) q[5];
rz(2.169547858835075) q[5];
ry(0.030199784099160517) q[6];
rz(-1.4213447175029594) q[6];
ry(3.087832057844572) q[7];
rz(1.8363816656039171) q[7];
ry(3.0229225366895762) q[8];
rz(1.1497205231963559) q[8];
ry(2.8745344896206673) q[9];
rz(0.8757471279549689) q[9];
ry(0.9139124094323074) q[10];
rz(-2.3087742962661224) q[10];
ry(1.1088510246194305) q[11];
rz(-2.3218580638433446) q[11];