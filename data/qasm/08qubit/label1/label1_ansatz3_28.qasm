OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-1.256368659623142) q[0];
rz(2.82634602068782) q[0];
ry(3.0193997811072792) q[1];
rz(-1.3527846862341877) q[1];
ry(1.3116176305850589) q[2];
rz(2.744326295545338) q[2];
ry(0.9346181766183631) q[3];
rz(2.839362369372065) q[3];
ry(-1.8827396233041496) q[4];
rz(0.14960973757401594) q[4];
ry(-2.8979908610987586) q[5];
rz(0.24243548066504075) q[5];
ry(-2.419927342365768) q[6];
rz(3.131860105706282) q[6];
ry(0.11353433020397229) q[7];
rz(1.9464525014126242) q[7];
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
ry(2.063456389945677) q[0];
rz(1.3801638433943795) q[0];
ry(2.583855873683375) q[1];
rz(1.6145395320430354) q[1];
ry(-2.7283054141332723) q[2];
rz(-0.40702000535051824) q[2];
ry(1.4663001724023472) q[3];
rz(1.68645545249804) q[3];
ry(2.5496842297743587) q[4];
rz(-0.6981939891995889) q[4];
ry(1.8481468351232158) q[5];
rz(-2.7086014368946887) q[5];
ry(-3.0512934534224367) q[6];
rz(-1.5735594838707083) q[6];
ry(0.6931581649206338) q[7];
rz(1.2794960672508733) q[7];
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
ry(-0.12725950231411703) q[0];
rz(-0.844162608441028) q[0];
ry(1.7265774157005387) q[1];
rz(-1.798005177624857) q[1];
ry(-0.599957836583475) q[2];
rz(-0.9911080182185685) q[2];
ry(-1.1164278535197898) q[3];
rz(0.8927659744883928) q[3];
ry(1.9847480081199105) q[4];
rz(2.377933371513004) q[4];
ry(1.2136895989601006) q[5];
rz(-0.32348359151652284) q[5];
ry(2.702487327016325) q[6];
rz(2.9313930287090484) q[6];
ry(-1.2193226618117687) q[7];
rz(2.4941030906553805) q[7];
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
ry(1.099578398067963) q[0];
rz(0.5350969982373168) q[0];
ry(3.0285111480114923) q[1];
rz(3.0067805357473114) q[1];
ry(-1.2398022375728455) q[2];
rz(1.7162831149706057) q[2];
ry(2.1552785011828908) q[3];
rz(3.0933517434818367) q[3];
ry(-0.32222023920543114) q[4];
rz(0.3614983895054174) q[4];
ry(2.3856637528609315) q[5];
rz(-0.9830549378248339) q[5];
ry(-1.5243495161861735) q[6];
rz(3.044490165926349) q[6];
ry(1.4280160183532782) q[7];
rz(-1.3962657439589483) q[7];
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
ry(3.054908750839314) q[0];
rz(-1.8739024234282706) q[0];
ry(-1.651747488545162) q[1];
rz(-1.3911810822640707) q[1];
ry(-2.4626637978559516) q[2];
rz(-2.2687400732995346) q[2];
ry(0.2534858655337313) q[3];
rz(0.7395395624568336) q[3];
ry(-2.742487259829561) q[4];
rz(-0.24163004482314676) q[4];
ry(-0.46132771534692285) q[5];
rz(2.3940847502833202) q[5];
ry(-1.9356578274637481) q[6];
rz(0.022617474587813824) q[6];
ry(0.8951012858870656) q[7];
rz(2.312016925658249) q[7];
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
ry(-1.5964315374823916) q[0];
rz(-0.17991094671686625) q[0];
ry(-1.0931659118546815) q[1];
rz(-3.0866738520395263) q[1];
ry(0.6198666995373312) q[2];
rz(-0.9438567065051541) q[2];
ry(-1.4869762155511665) q[3];
rz(0.6172658953012471) q[3];
ry(-2.715285645936141) q[4];
rz(-1.4758423568981496) q[4];
ry(-1.57052110815554) q[5];
rz(2.1039384541438766) q[5];
ry(2.916153719609753) q[6];
rz(2.4667474147480632) q[6];
ry(-2.3539537789479312) q[7];
rz(-0.6348475974625903) q[7];
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
ry(-1.8281673732875747) q[0];
rz(1.2062665518362354) q[0];
ry(-1.573767685911246) q[1];
rz(0.46704470604993814) q[1];
ry(0.83654429269985) q[2];
rz(-1.2663424747322294) q[2];
ry(-1.8521565070827055) q[3];
rz(2.039868958917062) q[3];
ry(-2.492996934504807) q[4];
rz(0.588581103523753) q[4];
ry(1.7686395848737009) q[5];
rz(2.6081248038544556) q[5];
ry(-1.4824597694751431) q[6];
rz(1.2557172997303931) q[6];
ry(1.6550534842778457) q[7];
rz(1.8359203601115504) q[7];
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
ry(1.488225276855669) q[0];
rz(-3.0353656368446402) q[0];
ry(0.41301686556413647) q[1];
rz(-2.086980992775296) q[1];
ry(1.9164668867861814) q[2];
rz(0.2708546243787771) q[2];
ry(1.8405149263167608) q[3];
rz(1.3447693621988488) q[3];
ry(1.2289191238188302) q[4];
rz(-1.787718460445403) q[4];
ry(0.2955044054358149) q[5];
rz(-2.748926876752398) q[5];
ry(-2.6474322365546086) q[6];
rz(2.504198526224069) q[6];
ry(-2.0600253797777466) q[7];
rz(2.846144936161617) q[7];
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
ry(1.7620839747947639) q[0];
rz(2.353945767315562) q[0];
ry(-2.153814068864267) q[1];
rz(-0.7991962117935367) q[1];
ry(-2.4678465743132065) q[2];
rz(-1.483285328451597) q[2];
ry(-1.287883778106221) q[3];
rz(1.3537782743066262) q[3];
ry(-0.9743434873402305) q[4];
rz(1.0551895306642847) q[4];
ry(-2.15935769276034) q[5];
rz(-2.2813143134714933) q[5];
ry(1.8652323025418571) q[6];
rz(-2.4943506838336513) q[6];
ry(0.2637417120763429) q[7];
rz(-1.3363193429594817) q[7];
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
ry(2.1558717819015976) q[0];
rz(2.630553307138793) q[0];
ry(2.793196964447857) q[1];
rz(-1.928101577481809) q[1];
ry(0.08609319417679961) q[2];
rz(-1.5951840239428128) q[2];
ry(-0.7433384641753076) q[3];
rz(1.5786973396914925) q[3];
ry(-2.257069709697554) q[4];
rz(-2.2719838054039383) q[4];
ry(0.1351533577887567) q[5];
rz(2.8502637454053574) q[5];
ry(-0.19333703289621873) q[6];
rz(-0.5647566162236349) q[6];
ry(0.4785567392237846) q[7];
rz(0.9980451347631895) q[7];
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
ry(1.6341839812151535) q[0];
rz(-1.112710539699099) q[0];
ry(2.915284738870665) q[1];
rz(0.5127312127161998) q[1];
ry(-0.33660656128435124) q[2];
rz(-2.8715975883767606) q[2];
ry(2.9553667466414475) q[3];
rz(1.0102643082166967) q[3];
ry(2.0179201659537633) q[4];
rz(-2.6212119335166664) q[4];
ry(-2.5890355395726456) q[5];
rz(0.3076427976046112) q[5];
ry(-1.505281570835722) q[6];
rz(-1.4563496017091273) q[6];
ry(-1.2141401420442701) q[7];
rz(-1.831277268946934) q[7];
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
ry(-1.328469034929894) q[0];
rz(-2.7988631934405537) q[0];
ry(-2.3145868145888775) q[1];
rz(0.36887574450762806) q[1];
ry(-1.345926146124187) q[2];
rz(-2.501900821441345) q[2];
ry(1.093220462155772) q[3];
rz(0.584873310875622) q[3];
ry(-0.6261294561705473) q[4];
rz(-0.34877887982946176) q[4];
ry(2.548840268412447) q[5];
rz(-1.7501543326127837) q[5];
ry(-0.27230260773381165) q[6];
rz(-2.1952745403858778) q[6];
ry(2.59886802440586) q[7];
rz(0.36982568777639363) q[7];
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
ry(1.3732183309934394) q[0];
rz(-1.1165717601636425) q[0];
ry(-1.334137152578296) q[1];
rz(2.2148564051887893) q[1];
ry(-0.6125837937535143) q[2];
rz(0.6340517508767025) q[2];
ry(-1.907976611058498) q[3];
rz(-0.7027794829296717) q[3];
ry(0.7377510481545242) q[4];
rz(-0.9136758167267337) q[4];
ry(0.4954192596135654) q[5];
rz(2.961215856075876) q[5];
ry(1.6457686277021633) q[6];
rz(2.622762188934265) q[6];
ry(-0.012722288400084382) q[7];
rz(2.517158975265674) q[7];
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
ry(2.533665282722561) q[0];
rz(-1.1507138108148594) q[0];
ry(0.17565348505827671) q[1];
rz(-1.781935800004228) q[1];
ry(3.046149945445757) q[2];
rz(1.6119192244586262) q[2];
ry(1.3369325231314066) q[3];
rz(1.9257253969742267) q[3];
ry(0.5588296118094682) q[4];
rz(-2.6058170416582485) q[4];
ry(2.894435240903773) q[5];
rz(0.3335502685870408) q[5];
ry(1.1845631061846145) q[6];
rz(0.9422428539887663) q[6];
ry(-0.04607200898284545) q[7];
rz(-2.9893645302811493) q[7];
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
ry(0.8570615219714632) q[0];
rz(-0.4285895782513043) q[0];
ry(1.7199333115862476) q[1];
rz(0.33227236315071) q[1];
ry(0.31388897321616216) q[2];
rz(2.3951635732209247) q[2];
ry(-2.3400966710186086) q[3];
rz(0.40104993435330805) q[3];
ry(-1.6563699098077107) q[4];
rz(-1.5685416169779387) q[4];
ry(2.0935159250957334) q[5];
rz(2.118354409414094) q[5];
ry(-2.620560318594619) q[6];
rz(0.9349958535277293) q[6];
ry(1.9293130468938386) q[7];
rz(-0.7964642960328535) q[7];
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
ry(2.6268619040352443) q[0];
rz(0.2509609267860032) q[0];
ry(1.1704026730646095) q[1];
rz(-1.9995007416301283) q[1];
ry(1.9644012535859572) q[2];
rz(0.20640021913950338) q[2];
ry(-0.9496472344675313) q[3];
rz(-3.0626504515766766) q[3];
ry(-0.6283303093786543) q[4];
rz(1.1426781841984457) q[4];
ry(0.9641733360214848) q[5];
rz(0.9619435530546594) q[5];
ry(0.7419066858898503) q[6];
rz(-0.295215566591131) q[6];
ry(0.8384693524786107) q[7];
rz(-2.2627695647200565) q[7];
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
ry(1.507285297511225) q[0];
rz(-1.298586345811029) q[0];
ry(2.2742813243828506) q[1];
rz(1.2485249006742976) q[1];
ry(-2.2430496832037248) q[2];
rz(3.0125840241830892) q[2];
ry(-2.1601083071162144) q[3];
rz(2.492795610291135) q[3];
ry(-1.410782217258603) q[4];
rz(-0.569229464195473) q[4];
ry(-2.2553508626980916) q[5];
rz(-2.342350074260579) q[5];
ry(-1.554340840182883) q[6];
rz(-3.123331183670819) q[6];
ry(-2.5966615686043806) q[7];
rz(-1.5361962169503869) q[7];
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
ry(-2.832609613831626) q[0];
rz(0.8773409770449074) q[0];
ry(2.3105744194850275) q[1];
rz(-2.7014042806122394) q[1];
ry(-3.0397860787082456) q[2];
rz(1.0504537924552348) q[2];
ry(-2.2102394066786264) q[3];
rz(-1.0326463819516294) q[3];
ry(-0.9576665148620663) q[4];
rz(-1.7023946628503492) q[4];
ry(1.345289805260732) q[5];
rz(1.2317303586310695) q[5];
ry(2.3205668692555474) q[6];
rz(-0.2649287833185313) q[6];
ry(2.3635769344954003) q[7];
rz(0.4862800854127208) q[7];
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
ry(-0.6841415266001479) q[0];
rz(0.6271948457844632) q[0];
ry(-1.1286029873490029) q[1];
rz(-1.1015991754757755) q[1];
ry(-2.2148594235286225) q[2];
rz(-1.0694353820826201) q[2];
ry(0.529716474556752) q[3];
rz(2.797097196272856) q[3];
ry(-2.6531332706147577) q[4];
rz(1.5810584538497539) q[4];
ry(-0.8906638280067946) q[5];
rz(1.4096315803101593) q[5];
ry(-0.4504639613481856) q[6];
rz(1.2958550951869634) q[6];
ry(-2.738097424526043) q[7];
rz(-0.17475972166493925) q[7];
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
ry(0.40752456992120994) q[0];
rz(2.8774181873646136) q[0];
ry(1.3565693551711646) q[1];
rz(1.6261198051447476) q[1];
ry(0.9640084240480556) q[2];
rz(-1.7064231399354126) q[2];
ry(1.1881360826842773) q[3];
rz(0.09546853747969462) q[3];
ry(0.8426176881067029) q[4];
rz(3.075448258508821) q[4];
ry(0.229458326575112) q[5];
rz(1.1144094628193688) q[5];
ry(2.604903469526554) q[6];
rz(-0.8117574817012154) q[6];
ry(-0.46092455043218855) q[7];
rz(1.0963063867344869) q[7];
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
ry(0.08646109443918375) q[0];
rz(-1.199423135751612) q[0];
ry(-2.195073520157102) q[1];
rz(-0.9191973237829993) q[1];
ry(-1.324532083538955) q[2];
rz(-0.4126195082013985) q[2];
ry(0.4212701608007975) q[3];
rz(0.29260862247777103) q[3];
ry(-0.16945102537020293) q[4];
rz(-0.35744767699304847) q[4];
ry(-1.5672365889479911) q[5];
rz(-0.14272386950463642) q[5];
ry(-2.3335141541257625) q[6];
rz(-0.24073836988563713) q[6];
ry(1.4273322828335284) q[7];
rz(0.11172561663047356) q[7];
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
ry(2.3699849653001754) q[0];
rz(0.08471754160051323) q[0];
ry(-0.17405187631913854) q[1];
rz(1.8278579747979649) q[1];
ry(2.82679968072334) q[2];
rz(0.285294391382978) q[2];
ry(1.2135928783247518) q[3];
rz(2.3228442291356326) q[3];
ry(-0.6452161032826509) q[4];
rz(-0.043268556766308926) q[4];
ry(-2.5621310693658277) q[5];
rz(2.244885346352828) q[5];
ry(1.2159670993291998) q[6];
rz(1.0158948804792898) q[6];
ry(2.0324030606849286) q[7];
rz(-3.045771756538731) q[7];
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
ry(0.29276470299997076) q[0];
rz(-2.4151635268014005) q[0];
ry(0.7788094967315512) q[1];
rz(3.0164465165939243) q[1];
ry(-2.6339144826681307) q[2];
rz(0.663461727301751) q[2];
ry(2.432742713960938) q[3];
rz(0.22731648321674225) q[3];
ry(-2.01760655825701) q[4];
rz(-0.3778368048089087) q[4];
ry(1.9442530058198186) q[5];
rz(-3.1003064987114612) q[5];
ry(-0.10157575702063415) q[6];
rz(-0.9935488662587172) q[6];
ry(-0.7016483299114222) q[7];
rz(2.0474252121827177) q[7];
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
ry(-0.9938451176302543) q[0];
rz(2.3623405058460167) q[0];
ry(2.3540290247613225) q[1];
rz(-0.5538116222249848) q[1];
ry(-2.1966624192721644) q[2];
rz(1.958941243560497) q[2];
ry(-2.476110617239582) q[3];
rz(0.2375762547575448) q[3];
ry(2.4009123922566658) q[4];
rz(-2.0396562021053666) q[4];
ry(-2.239872434772093) q[5];
rz(-0.5783629624921116) q[5];
ry(1.707854854112705) q[6];
rz(-0.3050159034676732) q[6];
ry(-2.47361752269314) q[7];
rz(-0.0002833172155165542) q[7];
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
ry(1.7073275872002467) q[0];
rz(-0.4875912893827872) q[0];
ry(-0.6608876686772454) q[1];
rz(0.7752482483534839) q[1];
ry(1.1875292942250848) q[2];
rz(-0.5066848354859337) q[2];
ry(-0.6777903993102284) q[3];
rz(2.8091201532646877) q[3];
ry(2.3025678094085755) q[4];
rz(2.34991182397044) q[4];
ry(0.3096880500981518) q[5];
rz(1.5977960881768025) q[5];
ry(1.3925530079152502) q[6];
rz(-2.45040573431753) q[6];
ry(0.7910765137481395) q[7];
rz(3.026322494153328) q[7];
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
ry(1.7634194209600238) q[0];
rz(0.7543686271899854) q[0];
ry(-1.0041898911541074) q[1];
rz(2.6347093736276395) q[1];
ry(-0.6457819882443725) q[2];
rz(2.8868088569429204) q[2];
ry(-2.3225027354856818) q[3];
rz(1.1508589439795998) q[3];
ry(2.9810627525706845) q[4];
rz(2.7076515432322936) q[4];
ry(2.5009347068392023) q[5];
rz(-2.015017678044842) q[5];
ry(-2.6752499639149185) q[6];
rz(1.4019584209546911) q[6];
ry(-1.5465308460629776) q[7];
rz(2.3368439605658353) q[7];
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
ry(0.7188594625312329) q[0];
rz(-1.192173567558569) q[0];
ry(-2.82633915659677) q[1];
rz(-2.652774734442638) q[1];
ry(-0.8162946615826682) q[2];
rz(0.056906675821120516) q[2];
ry(-0.9917503010382154) q[3];
rz(-0.24390602200073896) q[3];
ry(2.751466326406098) q[4];
rz(2.5766994999913524) q[4];
ry(2.7952543680811544) q[5];
rz(-2.870080997390148) q[5];
ry(2.2786493629660294) q[6];
rz(0.4052789111683988) q[6];
ry(1.541594243493366) q[7];
rz(0.38235941773537974) q[7];
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
ry(-1.784866209948061) q[0];
rz(1.0075660469199406) q[0];
ry(0.5246823442684209) q[1];
rz(1.2937573380100647) q[1];
ry(-0.49500526353776914) q[2];
rz(2.329118381065873) q[2];
ry(2.3327707148056565) q[3];
rz(2.1238546835087653) q[3];
ry(-1.2247009500122523) q[4];
rz(1.002874961990818) q[4];
ry(2.618335371439844) q[5];
rz(2.4153858663326746) q[5];
ry(1.3937222235943565) q[6];
rz(-0.22901737044825407) q[6];
ry(-0.46469127982093833) q[7];
rz(0.25855046108307916) q[7];
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
ry(-2.4754281521345933) q[0];
rz(-0.8242334521805672) q[0];
ry(0.4859383484257703) q[1];
rz(0.008967793746783137) q[1];
ry(-1.4922383341969079) q[2];
rz(-2.6974900865691556) q[2];
ry(0.3994784919417391) q[3];
rz(0.7386046081154214) q[3];
ry(-1.0727308388331167) q[4];
rz(1.5443103093904518) q[4];
ry(-2.4974081956557095) q[5];
rz(1.8206628171024788) q[5];
ry(-0.5520706890691242) q[6];
rz(-1.5235049612865819) q[6];
ry(2.1980584180032654) q[7];
rz(-1.5481145022012957) q[7];
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
ry(0.532965746682313) q[0];
rz(2.330294081416761) q[0];
ry(1.0371898788268825) q[1];
rz(-2.0374958705595194) q[1];
ry(-2.2863402228787675) q[2];
rz(-2.2642324129494114) q[2];
ry(-2.575133499185081) q[3];
rz(0.6274710329562304) q[3];
ry(-0.9734337384125754) q[4];
rz(-3.0139702669952806) q[4];
ry(2.4178947786187983) q[5];
rz(1.6501984171735193) q[5];
ry(-0.8873861399794802) q[6];
rz(0.6633645461338382) q[6];
ry(-0.7523330458359228) q[7];
rz(-2.561521163553998) q[7];
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
ry(0.5742966879375542) q[0];
rz(-2.3434612288116727) q[0];
ry(-2.460986211849953) q[1];
rz(-2.12095865052936) q[1];
ry(0.827873512852916) q[2];
rz(2.4830515873544683) q[2];
ry(-1.142167674922283) q[3];
rz(-1.027319182216515) q[3];
ry(-2.1160137017618057) q[4];
rz(2.818075556834678) q[4];
ry(0.048325894666396224) q[5];
rz(2.9550586037172386) q[5];
ry(-1.1983061020417294) q[6];
rz(1.7205589994825723) q[6];
ry(-3.064116573732544) q[7];
rz(-0.11654283680537421) q[7];
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
ry(-1.6459725395081284) q[0];
rz(2.698593371174536) q[0];
ry(-2.0961596819448833) q[1];
rz(-1.4920710138903575) q[1];
ry(-0.8563524704626482) q[2];
rz(0.8367477382108945) q[2];
ry(1.0213662608898302) q[3];
rz(-1.9573694918097815) q[3];
ry(2.830090370883682) q[4];
rz(0.25821137152947055) q[4];
ry(-2.7619011410634444) q[5];
rz(2.0531475885807646) q[5];
ry(1.4344341891680115) q[6];
rz(2.7375175775432137) q[6];
ry(2.2553386711296635) q[7];
rz(-2.8995575088408536) q[7];