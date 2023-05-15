OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-0.5845072607548866) q[0];
rz(-0.34803487571495434) q[0];
ry(2.327879389141866) q[1];
rz(1.0723193585387767) q[1];
ry(1.905851204235372) q[2];
rz(-1.2184485718660334) q[2];
ry(-0.634949036772667) q[3];
rz(-2.0758553148502568) q[3];
ry(-2.4487983070563843) q[4];
rz(-1.8811381277750021) q[4];
ry(1.095009876790014) q[5];
rz(-3.018794395546803) q[5];
ry(-3.1297554412602833) q[6];
rz(-1.2789629899267965) q[6];
ry(-0.8110082576953326) q[7];
rz(-1.736397697934554) q[7];
ry(-0.4653301448264884) q[8];
rz(-1.4425047089087855) q[8];
ry(-1.8538109539167993) q[9];
rz(2.997680765577156) q[9];
ry(-1.6655401039873323) q[10];
rz(0.24422834237590507) q[10];
ry(1.5807058454255056) q[11];
rz(2.7110091803992304) q[11];
ry(0.0603926607616413) q[12];
rz(1.9935542387463097) q[12];
ry(2.0433347473903387) q[13];
rz(-1.1243763983257482) q[13];
ry(1.078828201029025) q[14];
rz(-2.8209712640237257) q[14];
ry(-0.07122813858668309) q[15];
rz(-1.9583616753596518) q[15];
ry(2.9518553697059837) q[16];
rz(-3.022304347517121) q[16];
ry(-0.021520406945080044) q[17];
rz(2.486045696057888) q[17];
ry(0.6803540527558809) q[18];
rz(-2.859948658512771) q[18];
ry(-2.0765538646183224) q[19];
rz(-2.3470358236861877) q[19];
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
ry(0.18570089599775194) q[0];
rz(-0.23479850111924172) q[0];
ry(0.019069285571717096) q[1];
rz(0.6433876090248054) q[1];
ry(-3.131745722521183) q[2];
rz(-1.3137491732770419) q[2];
ry(-3.1025007738200205) q[3];
rz(1.558559268551507) q[3];
ry(-0.07226303639418397) q[4];
rz(1.8323688079629754) q[4];
ry(0.15882968704130374) q[5];
rz(0.18029431678020258) q[5];
ry(2.2843439602209474) q[6];
rz(-0.35891602540671713) q[6];
ry(-1.806043171872889) q[7];
rz(-2.0344499994342087) q[7];
ry(-1.205033780103662) q[8];
rz(-2.629540952184907) q[8];
ry(1.3681778933448532) q[9];
rz(-2.918482627393914) q[9];
ry(-2.021919094930695) q[10];
rz(3.0770762684636246) q[10];
ry(-2.490715368974578) q[11];
rz(2.411055646821335) q[11];
ry(0.8723073199914442) q[12];
rz(-1.2689461928065624) q[12];
ry(3.114973852185619) q[13];
rz(-2.024754916535967) q[13];
ry(3.110684729206963) q[14];
rz(-0.6866476478073161) q[14];
ry(-1.5977923797499716) q[15];
rz(-1.3922826397965438) q[15];
ry(-0.16523001651450286) q[16];
rz(2.802803154253744) q[16];
ry(0.03926743957482752) q[17];
rz(-1.2705491587609583) q[17];
ry(1.7437166152410817) q[18];
rz(-1.281051058687794) q[18];
ry(-1.4293673802600075) q[19];
rz(-0.9271677635845915) q[19];
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
ry(2.6453308932406556) q[0];
rz(-2.9724081608492336) q[0];
ry(-1.058426083809267) q[1];
rz(-0.36464670686285494) q[1];
ry(-1.1280366574532101) q[2];
rz(-2.2422812541354107) q[2];
ry(-0.10442976142548321) q[3];
rz(-2.052763407640823) q[3];
ry(-2.218539973551347) q[4];
rz(1.248898559595999) q[4];
ry(3.0497757437234894) q[5];
rz(1.1753511486989998) q[5];
ry(3.1125068719529407) q[6];
rz(-0.41477529407915004) q[6];
ry(-0.003542343309986151) q[7];
rz(-2.010208383381837) q[7];
ry(2.0179478177036856) q[8];
rz(-1.1401227165293768) q[8];
ry(3.099143339691731) q[9];
rz(2.72466456565393) q[9];
ry(0.31033079035172495) q[10];
rz(-2.9998265084610183) q[10];
ry(-0.9232311740448491) q[11];
rz(-0.14810755701533812) q[11];
ry(-0.04154325094067879) q[12];
rz(-1.522961938783545) q[12];
ry(-0.5920811260832332) q[13];
rz(1.071975412085406) q[13];
ry(-2.9328564241685156) q[14];
rz(-1.1942966508907475) q[14];
ry(-2.571667628982283) q[15];
rz(1.8588515415674696) q[15];
ry(2.8548027458843697) q[16];
rz(-0.2832750090178857) q[16];
ry(3.0816954696372982) q[17];
rz(-2.230546199640097) q[17];
ry(-1.341809947186936) q[18];
rz(-0.8215971391095476) q[18];
ry(1.2225010773204996) q[19];
rz(-2.8596896665933245) q[19];
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
ry(-0.3552522903348141) q[0];
rz(2.1209797789243616) q[0];
ry(0.02705648062284129) q[1];
rz(0.18468179727061546) q[1];
ry(3.1171748806541255) q[2];
rz(-2.6450544420185285) q[2];
ry(-3.1288080241449454) q[3];
rz(2.3803096691459458) q[3];
ry(-2.8311572258348625) q[4];
rz(-0.594105521499579) q[4];
ry(2.52706089785415) q[5];
rz(0.5593495536747604) q[5];
ry(2.2809629629981467) q[6];
rz(0.009958615761443341) q[6];
ry(-1.040213371164186) q[7];
rz(-0.5966694125457781) q[7];
ry(-0.10898999344071164) q[8];
rz(1.214692252873685) q[8];
ry(-2.807797923505561) q[9];
rz(-2.093939890476933) q[9];
ry(0.10752053227736624) q[10];
rz(2.237735082295881) q[10];
ry(-1.034393601131742) q[11];
rz(0.3837526417907015) q[11];
ry(1.3461981315523905) q[12];
rz(2.7340419494230925) q[12];
ry(-3.0050125514295734) q[13];
rz(3.085719628389635) q[13];
ry(0.13369492844982211) q[14];
rz(2.6638012088135294) q[14];
ry(-1.7466955349903603) q[15];
rz(-1.0100300498473098) q[15];
ry(-2.4078834236821005) q[16];
rz(0.03341331312728713) q[16];
ry(0.0009276615777616943) q[17];
rz(-0.48527470324270267) q[17];
ry(0.3541105349411646) q[18];
rz(1.1659742979796845) q[18];
ry(-1.9022326216348784) q[19];
rz(1.480753825578474) q[19];
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
ry(1.83822330369975) q[0];
rz(-2.2585923946067514) q[0];
ry(-3.026012566318858) q[1];
rz(-2.054090625498925) q[1];
ry(-2.9523879585451294) q[2];
rz(-0.3730317259370568) q[2];
ry(2.324396840809313) q[3];
rz(1.3857789189172454) q[3];
ry(-1.4936881391703647) q[4];
rz(1.6848384809342447) q[4];
ry(2.3268985097913855) q[5];
rz(-1.5047125543270743) q[5];
ry(2.000400101432361) q[6];
rz(-2.0772321995019407) q[6];
ry(-1.6597815154505655) q[7];
rz(-2.8053570204504963) q[7];
ry(-1.1529486551031871) q[8];
rz(0.1458323966212447) q[8];
ry(-3.128506470440183) q[9];
rz(-1.1695882695718314) q[9];
ry(-2.985386873941061) q[10];
rz(-1.0411445544165812) q[10];
ry(-3.138274039619731) q[11];
rz(0.490616360862429) q[11];
ry(1.6391892138386182) q[12];
rz(-1.2367735482111575) q[12];
ry(1.3579259937652215) q[13];
rz(1.3798742648727265) q[13];
ry(-0.8972068528224879) q[14];
rz(-2.431660980247466) q[14];
ry(-3.1308164064514887) q[15];
rz(1.78435720269168) q[15];
ry(-0.8076874699522393) q[16];
rz(-2.4634401087796585) q[16];
ry(1.1837096144131678) q[17];
rz(2.455222807758039) q[17];
ry(2.788559980668012) q[18];
rz(-1.52665787837196) q[18];
ry(2.848420926987872) q[19];
rz(2.0812660545640838) q[19];
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
ry(-1.893806031328551) q[0];
rz(1.850127934413414) q[0];
ry(-0.005476981326008712) q[1];
rz(-2.654105607944115) q[1];
ry(-0.009538381783644202) q[2];
rz(-2.0230314729389294) q[2];
ry(0.7318285856756295) q[3];
rz(-1.951214012160797) q[3];
ry(-2.8017751422312576) q[4];
rz(-1.4226089359418692) q[4];
ry(3.1046512101285315) q[5];
rz(2.4167525098976483) q[5];
ry(3.1408488803584467) q[6];
rz(-1.0846059437433757) q[6];
ry(-0.8631237882734251) q[7];
rz(2.585842927222425) q[7];
ry(-2.1923927908201835) q[8];
rz(-0.5428722392903156) q[8];
ry(-0.31429687761291686) q[9];
rz(-0.6388963317988572) q[9];
ry(-1.3410843991318893) q[10];
rz(0.7158354285708215) q[10];
ry(0.2116636085470898) q[11];
rz(2.9302497612717313) q[11];
ry(3.00761401300428) q[12];
rz(-1.2588538440850705) q[12];
ry(-0.8026689578455861) q[13];
rz(-0.48235030502161275) q[13];
ry(-2.139068613282662) q[14];
rz(-2.4133310348262684) q[14];
ry(0.40873017902329745) q[15];
rz(-2.8948349717398285) q[15];
ry(-3.0896470716484115) q[16];
rz(-1.003870661988029) q[16];
ry(-0.06513028376395358) q[17];
rz(-2.490431799269763) q[17];
ry(-0.02923288482741356) q[18];
rz(0.5663015633196526) q[18];
ry(-1.5544693468834438) q[19];
rz(2.4064088133002604) q[19];
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
ry(-1.4768472893011984) q[0];
rz(-2.254904796423964) q[0];
ry(2.0937926254395616) q[1];
rz(-2.0260375815511527) q[1];
ry(3.1195337833147536) q[2];
rz(-1.6319238443452337) q[2];
ry(-0.18863221911881567) q[3];
rz(1.9332260509418804) q[3];
ry(2.153981300169156) q[4];
rz(0.8216893465425956) q[4];
ry(0.39840118638973865) q[5];
rz(0.11130022808211848) q[5];
ry(-1.317445685974375) q[6];
rz(-0.22598110922225167) q[6];
ry(2.6403308370137277) q[7];
rz(0.40519371935948933) q[7];
ry(-2.801187956723799) q[8];
rz(-0.6809382081528987) q[8];
ry(-2.8543202955672773) q[9];
rz(0.19729837991004914) q[9];
ry(-3.0136537343675514) q[10];
rz(-2.7361495707202126) q[10];
ry(-1.89679052655831) q[11];
rz(-2.942014353841127) q[11];
ry(-0.13842538160823017) q[12];
rz(0.2674707216269985) q[12];
ry(0.16517886392421183) q[13];
rz(3.112583117609016) q[13];
ry(-2.4613317617171027) q[14];
rz(-0.2716916911073639) q[14];
ry(2.2209219349751788) q[15];
rz(-1.7082099713109322) q[15];
ry(2.028495536660962) q[16];
rz(-2.601557868584426) q[16];
ry(-2.0654096280189727) q[17];
rz(-2.999550237466944) q[17];
ry(2.4039881666602216) q[18];
rz(-2.8487363496719103) q[18];
ry(-2.1712337110611353) q[19];
rz(0.8562912453607422) q[19];
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
ry(-1.1597033733630577) q[0];
rz(1.6807918692983348) q[0];
ry(3.043289796305997) q[1];
rz(-2.9575704086703234) q[1];
ry(0.08124545616035667) q[2];
rz(-1.047334013269016) q[2];
ry(2.3783555278572166) q[3];
rz(-1.8301728322647783) q[3];
ry(-1.0876637908789109) q[4];
rz(-1.2308836285722495) q[4];
ry(2.188570937948192) q[5];
rz(2.4871392036770117) q[5];
ry(2.598483629226234) q[6];
rz(-0.8559907208113787) q[6];
ry(-2.0860499433023616) q[7];
rz(-1.5593901791916107) q[7];
ry(2.868043475643109) q[8];
rz(-2.4438574551967673) q[8];
ry(0.23617836855323304) q[9];
rz(1.7332230373713804) q[9];
ry(2.0062106979350487) q[10];
rz(1.9729991459498617) q[10];
ry(-2.845680441768153) q[11];
rz(0.530949791567763) q[11];
ry(-3.010800921686467) q[12];
rz(-2.629630648430315) q[12];
ry(-0.8950444145872836) q[13];
rz(-2.640192280631492) q[13];
ry(-2.620452849621458) q[14];
rz(0.6569494139949487) q[14];
ry(-1.6990610928690035) q[15];
rz(2.5649331218317384) q[15];
ry(1.5060132399789827) q[16];
rz(-2.8540499441236316) q[16];
ry(-3.002545420463291) q[17];
rz(-0.6548708332734651) q[17];
ry(0.9225156022369809) q[18];
rz(-2.6136085901043424) q[18];
ry(1.548552333150797) q[19];
rz(1.7365684171155333) q[19];
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
ry(1.753885812626045) q[0];
rz(0.7531260668333699) q[0];
ry(-0.6744069123740675) q[1];
rz(0.3935382492729334) q[1];
ry(0.10882121328309059) q[2];
rz(-0.3120529306499973) q[2];
ry(-0.0636462143492489) q[3];
rz(1.742890135479691) q[3];
ry(-1.165551149455077) q[4];
rz(-2.287115050871419) q[4];
ry(-2.9371796061525433) q[5];
rz(-0.1722870556505357) q[5];
ry(-0.08220111323495059) q[6];
rz(2.767961153769944) q[6];
ry(3.0808266953824517) q[7];
rz(0.26578968441867756) q[7];
ry(0.38321113369960175) q[8];
rz(-0.9468048479409075) q[8];
ry(-3.127770099474919) q[9];
rz(0.14301980765101963) q[9];
ry(0.08708919715195051) q[10];
rz(-1.713317641067098) q[10];
ry(-2.8059985779339893) q[11];
rz(3.1389202372839713) q[11];
ry(-0.20261889247552872) q[12];
rz(-0.6662443899242189) q[12];
ry(2.8078359375457675) q[13];
rz(-1.304227824160162) q[13];
ry(2.269445016017917) q[14];
rz(-0.20426613048032827) q[14];
ry(-2.860565659319976) q[15];
rz(2.6697986201820685) q[15];
ry(-2.9589347596713838) q[16];
rz(0.04237339029131398) q[16];
ry(-1.2661308817553492) q[17];
rz(0.420336013682885) q[17];
ry(-0.13175341129329077) q[18];
rz(2.6118323485649855) q[18];
ry(-0.3816263190686092) q[19];
rz(-2.2369601020779735) q[19];
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
ry(-1.827748214571609) q[0];
rz(3.079992148773426) q[0];
ry(0.01817364606662064) q[1];
rz(-0.6796671862336945) q[1];
ry(-1.1447546470441932) q[2];
rz(-2.8633846936177805) q[2];
ry(-2.904364151381192) q[3];
rz(-1.0486005885893548) q[3];
ry(0.118144368076466) q[4];
rz(2.956356787951353) q[4];
ry(-1.3394881890133998) q[5];
rz(1.0494392329035873) q[5];
ry(1.15784957621193) q[6];
rz(-1.9520150567346368) q[6];
ry(2.855566166492123) q[7];
rz(0.8812226374214394) q[7];
ry(0.23701474248290832) q[8];
rz(0.8668975193862093) q[8];
ry(-3.0907563147518964) q[9];
rz(-1.7660794493277738) q[9];
ry(0.8183191019253346) q[10];
rz(2.6831370929163754) q[10];
ry(-0.26227221493389674) q[11];
rz(-2.8788815944607) q[11];
ry(-0.10077356118200521) q[12];
rz(-0.9619147497092361) q[12];
ry(0.002983933795220217) q[13];
rz(1.1458393546056407) q[13];
ry(-2.7982164063193533) q[14];
rz(0.07456089890665961) q[14];
ry(-1.4223076721093375) q[15];
rz(-2.3800321885424056) q[15];
ry(0.08807476943560985) q[16];
rz(-1.8485246104799316) q[16];
ry(-0.04520853361173001) q[17];
rz(0.4315028061019577) q[17];
ry(3.0421846077212304) q[18];
rz(2.842875937550448) q[18];
ry(-1.8226064789413563) q[19];
rz(-2.602690478505232) q[19];
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
ry(0.6770776408050665) q[0];
rz(2.9351365493118067) q[0];
ry(-0.0855438368993031) q[1];
rz(0.10015063801824502) q[1];
ry(0.11768798854023908) q[2];
rz(-0.4751389783398341) q[2];
ry(3.01621473795416) q[3];
rz(-2.526074642743826) q[3];
ry(1.1906193109970475) q[4];
rz(-2.8589206163993204) q[4];
ry(-1.7008895935180066) q[5];
rz(1.6655522682130126) q[5];
ry(0.1331922133523351) q[6];
rz(2.726229193979383) q[6];
ry(0.008929319724640727) q[7];
rz(-0.5018478124341308) q[7];
ry(-0.31039357012151153) q[8];
rz(2.677227993960086) q[8];
ry(-0.2486344229512811) q[9];
rz(-0.08187103904512226) q[9];
ry(2.592313055750842) q[10];
rz(2.917721572825894) q[10];
ry(1.2662467341950947) q[11];
rz(1.3231323655444727) q[11];
ry(2.961914062736824) q[12];
rz(1.2927080338544474) q[12];
ry(-0.4035468088877454) q[13];
rz(-1.0282043659194224) q[13];
ry(-1.4915947122186575) q[14];
rz(-2.9148447588348843) q[14];
ry(-0.15008187829786834) q[15];
rz(0.37992943796287704) q[15];
ry(-0.8205454262921483) q[16];
rz(-3.1200979527091843) q[16];
ry(-0.9382494763112021) q[17];
rz(1.4033679105644257) q[17];
ry(-1.5736237715785868) q[18];
rz(1.611674850844505) q[18];
ry(-0.2529480455389859) q[19];
rz(-2.623915396601838) q[19];
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
ry(-1.638935997877713) q[0];
rz(-1.0317447959646802) q[0];
ry(0.6132917755531552) q[1];
rz(1.8353482962631966) q[1];
ry(-2.229588448747251) q[2];
rz(-1.337018543837303) q[2];
ry(2.4553630620984355) q[3];
rz(-1.1200401914283784) q[3];
ry(3.0858871031327566) q[4];
rz(-2.269482768568605) q[4];
ry(-2.9235429421261405) q[5];
rz(2.932900044696152) q[5];
ry(-0.4039740186882623) q[6];
rz(2.914909811337329) q[6];
ry(1.756804883176357) q[7];
rz(2.1657508083284367) q[7];
ry(-1.8593166265864567) q[8];
rz(0.945939233189219) q[8];
ry(-1.599203917625205) q[9];
rz(1.7660604901994255) q[9];
ry(1.5531702275966581) q[10];
rz(2.443918757028362) q[10];
ry(-0.11803604335197448) q[11];
rz(-1.5570674121749173) q[11];
ry(-0.00826258818368153) q[12];
rz(0.6090386976297657) q[12];
ry(-0.2523290897420712) q[13];
rz(1.9101779613251) q[13];
ry(1.8223690859145298) q[14];
rz(-2.8526374998159576) q[14];
ry(-1.7115434199655581) q[15];
rz(-1.7224742028929416) q[15];
ry(3.096240218077903) q[16];
rz(1.4443776354748277) q[16];
ry(3.101409038474666) q[17];
rz(2.9363364133401415) q[17];
ry(-2.4114253276555786) q[18];
rz(1.7037797792163634) q[18];
ry(0.26590648796546784) q[19];
rz(2.3861361573152093) q[19];
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
ry(-1.421752164855008) q[0];
rz(1.8716539041491331) q[0];
ry(-1.3982167078305263) q[1];
rz(0.9084212725344467) q[1];
ry(0.4180425091752591) q[2];
rz(2.874369248853467) q[2];
ry(1.7782310760817506) q[3];
rz(2.2224733351713883) q[3];
ry(3.108790106125764) q[4];
rz(-2.047375004164129) q[4];
ry(-0.6042587252890677) q[5];
rz(3.1165746401868772) q[5];
ry(2.371014239546316) q[6];
rz(-3.0794616145553233) q[6];
ry(3.1396955371197657) q[7];
rz(1.654607580866648) q[7];
ry(3.0651116851233158) q[8];
rz(1.829500287879711) q[8];
ry(-3.0354844551995535) q[9];
rz(0.42854322041951703) q[9];
ry(0.044794377487170856) q[10];
rz(1.6785243459904144) q[10];
ry(-0.3293300101137093) q[11];
rz(0.5336367665627791) q[11];
ry(-2.7807432670862298) q[12];
rz(-0.40219552767191225) q[12];
ry(0.6343630045348725) q[13];
rz(-0.2953451240938351) q[13];
ry(-0.38821024609224697) q[14];
rz(-2.9007278126880713) q[14];
ry(-0.10146713046415788) q[15];
rz(0.49857416308188224) q[15];
ry(-1.5899222254821155) q[16];
rz(1.5992947570217688) q[16];
ry(2.9758637922479965) q[17];
rz(-3.138097078990421) q[17];
ry(2.9167898422949956) q[18];
rz(-0.34317036584979027) q[18];
ry(3.1290607590393873) q[19];
rz(-2.82071352936588) q[19];
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
ry(0.7368999397041879) q[0];
rz(0.4610919256588292) q[0];
ry(-2.299971172979811) q[1];
rz(-1.7926165889797239) q[1];
ry(3.1195511129099183) q[2];
rz(0.8277192677621397) q[2];
ry(-0.2316849444280119) q[3];
rz(-0.1574787885718535) q[3];
ry(-0.6701446924680745) q[4];
rz(2.423429151278411) q[4];
ry(2.8612546374818333) q[5];
rz(-1.8953199045791107) q[5];
ry(-1.30020641396806) q[6];
rz(-3.077455936149685) q[6];
ry(2.673049879657228) q[7];
rz(2.888571132849203) q[7];
ry(1.0587604118281018) q[8];
rz(1.9706041951538378) q[8];
ry(-1.9857396186693004) q[9];
rz(-0.9456492466637797) q[9];
ry(0.6137918254153103) q[10];
rz(2.2127880580612853) q[10];
ry(2.3915760388171567) q[11];
rz(-0.7857545118877178) q[11];
ry(0.03367605804562828) q[12];
rz(-0.6018920019125898) q[12];
ry(1.3333789911416378) q[13];
rz(-0.04801928502890527) q[13];
ry(2.990161685129841) q[14];
rz(-0.4339520759614346) q[14];
ry(-1.008041375588574) q[15];
rz(2.596454698519924) q[15];
ry(3.120868854988884) q[16];
rz(-3.0116056206432553) q[16];
ry(-0.08822569939027146) q[17];
rz(0.6855901618408681) q[17];
ry(1.1412081912155148) q[18];
rz(-2.3935433254464376) q[18];
ry(-1.9001014379118484) q[19];
rz(-0.417536662714491) q[19];
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
ry(-1.7464683415024789) q[0];
rz(1.895585255381761) q[0];
ry(-1.9204093437120684) q[1];
rz(0.5411237623961673) q[1];
ry(-0.39010169222473845) q[2];
rz(-2.919836847814465) q[2];
ry(-3.118148504725988) q[3];
rz(1.6653307031844602) q[3];
ry(3.133599220863209) q[4];
rz(-0.38140510917508674) q[4];
ry(-2.9173128632937626) q[5];
rz(1.761440224619595) q[5];
ry(-2.4573767170744163) q[6];
rz(3.0453602593757565) q[6];
ry(-3.1381556942547477) q[7];
rz(0.14305402231500358) q[7];
ry(3.099760718319608) q[8];
rz(-1.2850065087032734) q[8];
ry(2.989520359978354) q[9];
rz(3.0985495190430448) q[9];
ry(-0.06194735367860904) q[10];
rz(0.7583285469424929) q[10];
ry(3.037674959895369) q[11];
rz(1.1225129217744394) q[11];
ry(-0.06843187745863852) q[12];
rz(1.4093089864899981) q[12];
ry(-2.5046292628190225) q[13];
rz(2.932301154989421) q[13];
ry(0.8348666394579838) q[14];
rz(2.9628953498568706) q[14];
ry(-0.0030931065359464426) q[15];
rz(0.1403271898138259) q[15];
ry(2.5942024789408435) q[16];
rz(1.9031817329165026) q[16];
ry(2.7301611502750536) q[17];
rz(-0.614545298818852) q[17];
ry(0.19201942934797067) q[18];
rz(1.7385600405419424) q[18];
ry(1.515074318471597) q[19];
rz(2.0363057153218627) q[19];
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
ry(0.06422381635482341) q[0];
rz(1.3603090593546163) q[0];
ry(1.1792931138082299) q[1];
rz(-1.4583868022506004) q[1];
ry(3.076667785293992) q[2];
rz(-0.6899697379962273) q[2];
ry(-1.7687939018702075) q[3];
rz(-3.047283352642897) q[3];
ry(-2.1976858389828484) q[4];
rz(-1.2043328395450004) q[4];
ry(2.012153952540701) q[5];
rz(0.5398809052420972) q[5];
ry(-1.6004322824284476) q[6];
rz(-0.17297872227118538) q[6];
ry(-1.6112228483904143) q[7];
rz(-0.8305922874737098) q[7];
ry(1.9880718018683208) q[8];
rz(-2.2187434991151216) q[8];
ry(1.1013094288201888) q[9];
rz(2.2667021118874287) q[9];
ry(0.18882552869483388) q[10];
rz(-3.011211888465799) q[10];
ry(0.3086896163733428) q[11];
rz(-2.3373679080224514) q[11];
ry(-0.8296856833735012) q[12];
rz(-3.0211107687323033) q[12];
ry(0.15501360209429205) q[13];
rz(0.5288202653081645) q[13];
ry(-0.4429085259770362) q[14];
rz(0.3915776655928793) q[14];
ry(-3.1211582598147265) q[15];
rz(-2.5117974282024935) q[15];
ry(1.584435357262068) q[16];
rz(1.7003145335422944) q[16];
ry(3.136529622637533) q[17];
rz(1.9170139379021212) q[17];
ry(-3.124150528148152) q[18];
rz(-1.362549616940475) q[18];
ry(1.2759510101824603) q[19];
rz(-2.8014362700411617) q[19];
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
ry(-2.54780017986523) q[0];
rz(-0.0590359226088108) q[0];
ry(1.1422475179752443) q[1];
rz(0.027149458705392863) q[1];
ry(-2.59966902318132) q[2];
rz(0.019228664116839767) q[2];
ry(1.5380213032525092) q[3];
rz(-1.5704622260613579) q[3];
ry(0.009113421545058712) q[4];
rz(0.30860402188512287) q[4];
ry(3.1096628675222) q[5];
rz(2.9645999061022272) q[5];
ry(-2.202125828455559) q[6];
rz(-1.721281995019016) q[6];
ry(0.02502488316060955) q[7];
rz(1.1435687457857167) q[7];
ry(-0.016864932664083503) q[8];
rz(-1.3016280377190235) q[8];
ry(2.8297028636562898) q[9];
rz(2.9008888792301444) q[9];
ry(-3.0928664639603745) q[10];
rz(0.16023639436686565) q[10];
ry(0.48377025427323106) q[11];
rz(0.6820329599584838) q[11];
ry(0.06966443264649147) q[12];
rz(2.7790644352503553) q[12];
ry(0.08210382771515784) q[13];
rz(-2.813750408184666) q[13];
ry(-0.9771448465182279) q[14];
rz(1.866570854999484) q[14];
ry(3.0895672703710355) q[15];
rz(-1.4509412270465019) q[15];
ry(-3.1104686994005104) q[16];
rz(0.10343261874135747) q[16];
ry(2.6347802385579775) q[17];
rz(-0.12229459562460843) q[17];
ry(1.5812115564208968) q[18];
rz(3.0089259439472626) q[18];
ry(-1.4492635274397436) q[19];
rz(0.3923617514921416) q[19];
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
ry(2.3745414403662863) q[0];
rz(-2.1793230929039273) q[0];
ry(-1.8073232810086841) q[1];
rz(-2.3572865626194917) q[1];
ry(-1.552529318737219) q[2];
rz(1.5631795767149574) q[2];
ry(1.5790018924226852) q[3];
rz(1.348075488342193) q[3];
ry(1.1400218790384329) q[4];
rz(2.4221326368315603) q[4];
ry(2.9876729668411617) q[5];
rz(-3.010951201305417) q[5];
ry(3.086520371623622) q[6];
rz(1.2680691036887144) q[6];
ry(-2.895429862851032) q[7];
rz(2.6701265311686555) q[7];
ry(2.876566831320191) q[8];
rz(-0.9587330678149822) q[8];
ry(2.2033194887456387) q[9];
rz(-3.1390078239328356) q[9];
ry(2.302745697407951) q[10];
rz(2.5226251268831943) q[10];
ry(-0.03006613158472238) q[11];
rz(-1.2100223368894785) q[11];
ry(-2.1041890923171542) q[12];
rz(-0.6188740355923238) q[12];
ry(1.3773301274937664) q[13];
rz(3.0576196376770657) q[13];
ry(2.290919386111524) q[14];
rz(-2.036744650339058) q[14];
ry(0.44257399544058185) q[15];
rz(2.20512658116518) q[15];
ry(1.5923610128453047) q[16];
rz(0.11645205073094239) q[16];
ry(-1.5529650752847468) q[17];
rz(1.4542051282875406) q[17];
ry(-1.4224205464220967) q[18];
rz(2.6300408086170552) q[18];
ry(-2.8419960689555643) q[19];
rz(1.0887381780016043) q[19];
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
ry(0.83122747970571) q[0];
rz(2.0207322237818985) q[0];
ry(3.1358472553146517) q[1];
rz(0.5469257987311461) q[1];
ry(-2.871674073710497) q[2];
rz(0.5223942116819725) q[2];
ry(0.09577917861087304) q[3];
rz(-2.9050674935834886) q[3];
ry(-0.007951516654556517) q[4];
rz(-1.1350687881363812) q[4];
ry(-0.07790129131657597) q[5];
rz(-0.16924216889471322) q[5];
ry(0.8654086871627276) q[6];
rz(2.221706168807191) q[6];
ry(0.00525833039925902) q[7];
rz(0.20032980639119527) q[7];
ry(0.19931192352349736) q[8];
rz(2.6024931735655623) q[8];
ry(2.0158258229923427) q[9];
rz(2.9582927005639017) q[9];
ry(3.0494906103364126) q[10];
rz(0.8853148738215407) q[10];
ry(-0.18564945677223954) q[11];
rz(2.0597541114222464) q[11];
ry(3.037929217243624) q[12];
rz(1.4290168659707874) q[12];
ry(-0.050243499422628374) q[13];
rz(-2.9974856541226274) q[13];
ry(-1.9561072566996538) q[14];
rz(-2.891668727258741) q[14];
ry(1.413218134244576) q[15];
rz(-0.009198456006553946) q[15];
ry(3.1020252490936686) q[16];
rz(3.104439720727822) q[16];
ry(2.8601548794784555) q[17];
rz(-0.07689786698803082) q[17];
ry(1.5365166915085877) q[18];
rz(0.056528220878442025) q[18];
ry(-0.6188234026323114) q[19];
rz(1.3444945863122815) q[19];
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
ry(2.7393282530452727) q[0];
rz(3.120898186666847) q[0];
ry(0.6189488826564981) q[1];
rz(-2.7581059622778823) q[1];
ry(-3.117264232661479) q[2];
rz(-1.0304912507294615) q[2];
ry(1.569788687655547) q[3];
rz(-0.14001359733595073) q[3];
ry(3.027017286770121) q[4];
rz(1.676740963837562) q[4];
ry(1.4367000319405125) q[5];
rz(2.2430755918436702) q[5];
ry(-2.988452911315325) q[6];
rz(-2.7371906779144166) q[6];
ry(-0.8169599097991731) q[7];
rz(-1.4496205617785671) q[7];
ry(2.991116704732052) q[8];
rz(1.870509583149616) q[8];
ry(-0.058350402561810455) q[9];
rz(0.2501076587824446) q[9];
ry(-0.058806802877665376) q[10];
rz(-2.567081460956074) q[10];
ry(1.5774047155760373) q[11];
rz(1.9106933109806576) q[11];
ry(1.595726719053792) q[12];
rz(-2.3007295026155563) q[12];
ry(-0.18122473212738832) q[13];
rz(-0.8394462222202643) q[13];
ry(3.1404362532457983) q[14];
rz(0.19626985986046908) q[14];
ry(3.0648532167183395) q[15];
rz(-3.1305073364085807) q[15];
ry(-0.017854983432489036) q[16];
rz(1.5911318918476365) q[16];
ry(1.5742622154754593) q[17];
rz(1.936893157810803) q[17];
ry(1.5336577576804051) q[18];
rz(0.04815979156716509) q[18];
ry(0.004938356450608694) q[19];
rz(0.19131580791966307) q[19];
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
ry(2.187389324570681) q[0];
rz(-2.3482362694555365) q[0];
ry(1.5639342523254434) q[1];
rz(1.5696216107380405) q[1];
ry(-0.6139067400010081) q[2];
rz(-1.5771378258480846) q[2];
ry(-3.0357381894883915) q[3];
rz(-0.14496754215205507) q[3];
ry(-0.003472101571605674) q[4];
rz(-2.7744370296898357) q[4];
ry(-0.023538519278307746) q[5];
rz(1.047408123703761) q[5];
ry(3.0157291742075927) q[6];
rz(-0.04485801028686525) q[6];
ry(3.1184073558150374) q[7];
rz(-1.0976736509659422) q[7];
ry(2.936675269935614) q[8];
rz(0.4408058850387136) q[8];
ry(1.2004043020583355) q[9];
rz(1.9906020556359527) q[9];
ry(0.07788822107165329) q[10];
rz(3.051109004336523) q[10];
ry(-3.0444134347343224) q[11];
rz(2.6689566677807557) q[11];
ry(-3.0765429729818594) q[12];
rz(2.0084852312431876) q[12];
ry(3.1139461722989044) q[13];
rz(0.8116658131805495) q[13];
ry(1.896448752237354) q[14];
rz(-1.601541545163413) q[14];
ry(-1.4093376142067502) q[15];
rz(-0.10271303679945869) q[15];
ry(3.1358994736569374) q[16];
rz(1.034882393951768) q[16];
ry(3.055996171354884) q[17];
rz(1.4556315442021488) q[17];
ry(-3.034337865097199) q[18];
rz(-1.514615842999243) q[18];
ry(-0.5705425492588425) q[19];
rz(2.9676431319107737) q[19];
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
ry(-0.0013848048015727699) q[0];
rz(1.2160473252560369) q[0];
ry(1.5730045984053076) q[1];
rz(2.9318739915299834) q[1];
ry(-1.5653716978394954) q[2];
rz(3.065898324818755) q[2];
ry(-1.5774345232496851) q[3];
rz(-1.416525714164584) q[3];
ry(-2.606211705545819) q[4];
rz(-1.7800025219786035) q[4];
ry(-1.1036438890270697) q[5];
rz(-1.8600679563565432) q[5];
ry(-1.812696679919049) q[6];
rz(0.8957657947078115) q[6];
ry(-0.07220825919696772) q[7];
rz(-2.077661211875252) q[7];
ry(-1.7212521338683207) q[8];
rz(2.9613078824163765) q[8];
ry(-1.6759479300005315) q[9];
rz(0.28686242474771645) q[9];
ry(2.0668482848966976) q[10];
rz(-2.7172539499655866) q[10];
ry(0.09524265954195893) q[11];
rz(-0.020137239492951764) q[11];
ry(2.560400232982643) q[12];
rz(2.6171194129785107) q[12];
ry(-1.2222621678833816) q[13];
rz(0.4470308143836439) q[13];
ry(1.9913538182727226) q[14];
rz(-1.9966296389633378) q[14];
ry(-2.999976021722206) q[15];
rz(-1.2043118286534409) q[15];
ry(3.1219158634322275) q[16];
rz(0.5642424136612467) q[16];
ry(3.1211269375172335) q[17];
rz(0.5936050116177904) q[17];
ry(-1.5782044431785776) q[18];
rz(2.6857981277670873) q[18];
ry(-3.140272860593118) q[19];
rz(-0.6648390011334357) q[19];