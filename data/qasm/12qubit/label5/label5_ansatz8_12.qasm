OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.0027735923104471607) q[0];
ry(1.3315075487792423) q[1];
cx q[0],q[1];
ry(-2.336969160456593) q[0];
ry(-0.5520347417138115) q[1];
cx q[0],q[1];
ry(2.016625407962418) q[2];
ry(-1.224826086936728) q[3];
cx q[2],q[3];
ry(0.47874040786695815) q[2];
ry(-0.6552548679926299) q[3];
cx q[2],q[3];
ry(1.2025232234488448) q[4];
ry(0.8024181828258544) q[5];
cx q[4],q[5];
ry(-2.3971531183953196) q[4];
ry(-2.7843920526408463) q[5];
cx q[4],q[5];
ry(-0.906778889261517) q[6];
ry(1.197115047941658) q[7];
cx q[6],q[7];
ry(-1.1025719809849894) q[6];
ry(-1.8052285598899294) q[7];
cx q[6],q[7];
ry(-2.0095715698668477) q[8];
ry(2.3889944768049007) q[9];
cx q[8],q[9];
ry(-2.715137482765953) q[8];
ry(-0.11366397107517746) q[9];
cx q[8],q[9];
ry(2.1138358864299205) q[10];
ry(-0.07475013432458688) q[11];
cx q[10],q[11];
ry(-1.4595512083277669) q[10];
ry(-0.23059569694227644) q[11];
cx q[10],q[11];
ry(-0.9238331826951569) q[0];
ry(-1.1742125066174272) q[2];
cx q[0],q[2];
ry(2.63961439411581) q[0];
ry(0.44156156854841216) q[2];
cx q[0],q[2];
ry(-2.8835999542913275) q[2];
ry(1.8902306971365945) q[4];
cx q[2],q[4];
ry(-2.794123047063646) q[2];
ry(1.9269891052295582) q[4];
cx q[2],q[4];
ry(0.37873419650133755) q[4];
ry(-2.497962545487772) q[6];
cx q[4],q[6];
ry(-1.007840566208074) q[4];
ry(2.136285127891042) q[6];
cx q[4],q[6];
ry(1.1107690398102066) q[6];
ry(-0.5234557532455061) q[8];
cx q[6],q[8];
ry(1.6007830069385385) q[6];
ry(-1.0187389609920041) q[8];
cx q[6],q[8];
ry(1.2743900432488513) q[8];
ry(-0.18329382055674925) q[10];
cx q[8],q[10];
ry(1.3524308040907034) q[8];
ry(1.7049647185162486) q[10];
cx q[8],q[10];
ry(-2.7409279200082364) q[1];
ry(-0.9265781962880586) q[3];
cx q[1],q[3];
ry(-3.1301103028403943) q[1];
ry(-1.6978174418470797) q[3];
cx q[1],q[3];
ry(0.11530679924355791) q[3];
ry(2.4401431384093435) q[5];
cx q[3],q[5];
ry(-0.4910509316713317) q[3];
ry(-2.105640400314231) q[5];
cx q[3],q[5];
ry(1.446816220438941) q[5];
ry(2.5756531960311846) q[7];
cx q[5],q[7];
ry(-1.5426447704419353) q[5];
ry(-0.7501436154338434) q[7];
cx q[5],q[7];
ry(0.8389285653547944) q[7];
ry(3.140448025188126) q[9];
cx q[7],q[9];
ry(-1.3868814180766156) q[7];
ry(1.786247807897184) q[9];
cx q[7],q[9];
ry(0.9449536758189704) q[9];
ry(-0.6322995839320821) q[11];
cx q[9],q[11];
ry(-1.9417619619296982) q[9];
ry(-1.5765330402567033) q[11];
cx q[9],q[11];
ry(-1.6070839798679506) q[0];
ry(2.907162774770883) q[1];
cx q[0],q[1];
ry(1.9560333675673505) q[0];
ry(1.9869639431134565) q[1];
cx q[0],q[1];
ry(-2.143015207411872) q[2];
ry(-1.0650884717598732) q[3];
cx q[2],q[3];
ry(-1.0186081922853616) q[2];
ry(1.8434103365160457) q[3];
cx q[2],q[3];
ry(2.6081580079130866) q[4];
ry(2.273350641623675) q[5];
cx q[4],q[5];
ry(2.4801995457877273) q[4];
ry(-1.860606479148468) q[5];
cx q[4],q[5];
ry(3.1086491143432786) q[6];
ry(-2.4509774383552214) q[7];
cx q[6],q[7];
ry(-0.5337801363739525) q[6];
ry(2.405465194670902) q[7];
cx q[6],q[7];
ry(-2.9531172985557763) q[8];
ry(0.37293882378287563) q[9];
cx q[8],q[9];
ry(0.33414619527436307) q[8];
ry(-0.6537237979391846) q[9];
cx q[8],q[9];
ry(0.7671992093169031) q[10];
ry(0.7726335304792294) q[11];
cx q[10],q[11];
ry(-2.0268316054761875) q[10];
ry(-1.895851542499187) q[11];
cx q[10],q[11];
ry(0.4832183813740949) q[0];
ry(-1.4515683108475135) q[2];
cx q[0],q[2];
ry(-2.148534546207391) q[0];
ry(0.9374426127120303) q[2];
cx q[0],q[2];
ry(-0.5718433364448027) q[2];
ry(-2.0001882366284893) q[4];
cx q[2],q[4];
ry(1.486816496040115) q[2];
ry(-2.4560209216210342) q[4];
cx q[2],q[4];
ry(0.2275435968178794) q[4];
ry(-2.6598124592228762) q[6];
cx q[4],q[6];
ry(-1.4717563931222744) q[4];
ry(-0.30706898633377744) q[6];
cx q[4],q[6];
ry(-2.6805541410460263) q[6];
ry(-0.3223610226857856) q[8];
cx q[6],q[8];
ry(-2.3565726861336818) q[6];
ry(2.1290333689687424) q[8];
cx q[6],q[8];
ry(1.591855865256276) q[8];
ry(-1.0738085411111886) q[10];
cx q[8],q[10];
ry(3.079263647893704) q[8];
ry(-1.554591570889789) q[10];
cx q[8],q[10];
ry(2.7009167144042476) q[1];
ry(2.030267685152668) q[3];
cx q[1],q[3];
ry(-1.4960458631212274) q[1];
ry(2.754236034108537) q[3];
cx q[1],q[3];
ry(0.4621255452311112) q[3];
ry(-1.2222022628399767) q[5];
cx q[3],q[5];
ry(-0.2011738786962835) q[3];
ry(-1.5894518608085024) q[5];
cx q[3],q[5];
ry(2.441346680202361) q[5];
ry(-1.6571055911008248) q[7];
cx q[5],q[7];
ry(2.449289644794602) q[5];
ry(-1.3441733226053187) q[7];
cx q[5],q[7];
ry(-1.3924546539247624) q[7];
ry(0.013776767740077913) q[9];
cx q[7],q[9];
ry(2.083598638403272) q[7];
ry(0.5916738545334788) q[9];
cx q[7],q[9];
ry(-1.0852609551661505) q[9];
ry(-2.163089479121375) q[11];
cx q[9],q[11];
ry(-1.0338984459448146) q[9];
ry(2.587564441684701) q[11];
cx q[9],q[11];
ry(1.8014077067550964) q[0];
ry(0.1502533347528816) q[1];
cx q[0],q[1];
ry(2.797672761518618) q[0];
ry(1.5497647315586063) q[1];
cx q[0],q[1];
ry(-1.1334739689489028) q[2];
ry(1.1384773560073695) q[3];
cx q[2],q[3];
ry(-3.0537503821541607) q[2];
ry(-2.145980915980582) q[3];
cx q[2],q[3];
ry(-0.49519332048300924) q[4];
ry(0.7656836947326591) q[5];
cx q[4],q[5];
ry(-2.593916919208513) q[4];
ry(-1.0561454957118928) q[5];
cx q[4],q[5];
ry(-1.5803127863721755) q[6];
ry(-2.536426317179914) q[7];
cx q[6],q[7];
ry(-1.0233808054297002) q[6];
ry(2.5949991939466375) q[7];
cx q[6],q[7];
ry(2.8054336347813504) q[8];
ry(2.754888462683744) q[9];
cx q[8],q[9];
ry(-2.598117540594436) q[8];
ry(-1.2962300011191246) q[9];
cx q[8],q[9];
ry(1.2383948013789723) q[10];
ry(0.9362262528939931) q[11];
cx q[10],q[11];
ry(-2.491260216683109) q[10];
ry(-1.7114589187983962) q[11];
cx q[10],q[11];
ry(2.5710012628661203) q[0];
ry(2.159907559641068) q[2];
cx q[0],q[2];
ry(2.4386720833196756) q[0];
ry(-2.67955529997809) q[2];
cx q[0],q[2];
ry(2.8194058125683616) q[2];
ry(-1.4869553657988073) q[4];
cx q[2],q[4];
ry(-2.154234147297168) q[2];
ry(-1.9246501621770618) q[4];
cx q[2],q[4];
ry(2.0500029655317187) q[4];
ry(0.6106441157495187) q[6];
cx q[4],q[6];
ry(1.8777039664833592) q[4];
ry(2.406730697589259) q[6];
cx q[4],q[6];
ry(-1.1114018832997248) q[6];
ry(-2.5455838850501573) q[8];
cx q[6],q[8];
ry(-0.6856920780087146) q[6];
ry(2.2770807187663156) q[8];
cx q[6],q[8];
ry(-1.0969831124675329) q[8];
ry(2.3534006426166814) q[10];
cx q[8],q[10];
ry(1.9726971562079216) q[8];
ry(1.0664047447640996) q[10];
cx q[8],q[10];
ry(1.5192079369804325) q[1];
ry(1.6981509153570853) q[3];
cx q[1],q[3];
ry(-0.274168501655323) q[1];
ry(1.4025528919323718) q[3];
cx q[1],q[3];
ry(1.9115934449723344) q[3];
ry(3.0567821757908713) q[5];
cx q[3],q[5];
ry(0.508997570211708) q[3];
ry(0.3780364608950034) q[5];
cx q[3],q[5];
ry(-2.4696125730345515) q[5];
ry(-2.741866071799776) q[7];
cx q[5],q[7];
ry(3.0229915431379193) q[5];
ry(-1.4737003539942535) q[7];
cx q[5],q[7];
ry(1.8820991301963566) q[7];
ry(-1.401751436955838) q[9];
cx q[7],q[9];
ry(-2.564350130561484) q[7];
ry(-2.0012779250403696) q[9];
cx q[7],q[9];
ry(-2.8726480261703324) q[9];
ry(2.653689060938079) q[11];
cx q[9],q[11];
ry(-1.310492450311619) q[9];
ry(-1.4090627894075691) q[11];
cx q[9],q[11];
ry(1.883195846839321) q[0];
ry(0.9653104299782649) q[1];
cx q[0],q[1];
ry(2.1607153052094445) q[0];
ry(1.6672379874820684) q[1];
cx q[0],q[1];
ry(2.0046589141337963) q[2];
ry(2.742394019857046) q[3];
cx q[2],q[3];
ry(0.27389251939930404) q[2];
ry(1.0038249137923065) q[3];
cx q[2],q[3];
ry(0.21934585643362609) q[4];
ry(-0.4978587676004465) q[5];
cx q[4],q[5];
ry(2.404854484624789) q[4];
ry(0.4427534811863025) q[5];
cx q[4],q[5];
ry(-0.4999290035034214) q[6];
ry(-3.0376140502400406) q[7];
cx q[6],q[7];
ry(-1.0299355889345598) q[6];
ry(2.7688562163234405) q[7];
cx q[6],q[7];
ry(-2.5922337126824897) q[8];
ry(-0.5181677904802195) q[9];
cx q[8],q[9];
ry(-1.9271116457230857) q[8];
ry(-1.6862442446217027) q[9];
cx q[8],q[9];
ry(-2.654190784388735) q[10];
ry(2.7377638857410105) q[11];
cx q[10],q[11];
ry(1.478439133542204) q[10];
ry(2.498033725690128) q[11];
cx q[10],q[11];
ry(-0.3677916912299664) q[0];
ry(-0.6375164775863633) q[2];
cx q[0],q[2];
ry(-1.6659041780859716) q[0];
ry(-2.6690167659088484) q[2];
cx q[0],q[2];
ry(2.6497546290991343) q[2];
ry(-0.2672244283323559) q[4];
cx q[2],q[4];
ry(2.488863536140553) q[2];
ry(0.6070102188079138) q[4];
cx q[2],q[4];
ry(0.3537103048343269) q[4];
ry(-2.887254426686521) q[6];
cx q[4],q[6];
ry(2.9551570103082074) q[4];
ry(-1.082964761332783) q[6];
cx q[4],q[6];
ry(1.6356818126764) q[6];
ry(-2.119422945700281) q[8];
cx q[6],q[8];
ry(2.406485538344909) q[6];
ry(-2.3826904695621156) q[8];
cx q[6],q[8];
ry(-1.4909271606326677) q[8];
ry(-0.18436352746171636) q[10];
cx q[8],q[10];
ry(0.6601506167156703) q[8];
ry(2.237193580635836) q[10];
cx q[8],q[10];
ry(1.445769489329002) q[1];
ry(2.9894733802681577) q[3];
cx q[1],q[3];
ry(-2.634470455642604) q[1];
ry(0.9456528895685157) q[3];
cx q[1],q[3];
ry(-2.06432765865061) q[3];
ry(0.7267436504210034) q[5];
cx q[3],q[5];
ry(2.194098685711187) q[3];
ry(-2.6038848613650147) q[5];
cx q[3],q[5];
ry(-2.783033616124813) q[5];
ry(-1.2474692274230943) q[7];
cx q[5],q[7];
ry(-1.1336639277696705) q[5];
ry(2.40162379179954) q[7];
cx q[5],q[7];
ry(2.1883690400737477) q[7];
ry(2.3510346698551463) q[9];
cx q[7],q[9];
ry(2.512654926042843) q[7];
ry(-2.726678445131454) q[9];
cx q[7],q[9];
ry(-2.389172691851077) q[9];
ry(2.2238730122898267) q[11];
cx q[9],q[11];
ry(-1.1593768980806312) q[9];
ry(-1.6044060703831438) q[11];
cx q[9],q[11];
ry(1.686541393248565) q[0];
ry(1.0680347164556652) q[1];
cx q[0],q[1];
ry(0.30842227318958315) q[0];
ry(-1.0037762013582536) q[1];
cx q[0],q[1];
ry(-0.820508539347263) q[2];
ry(1.3710604803155713) q[3];
cx q[2],q[3];
ry(-1.899750943564376) q[2];
ry(-1.3166644711604913) q[3];
cx q[2],q[3];
ry(-0.7449158715279093) q[4];
ry(-0.8042371504389401) q[5];
cx q[4],q[5];
ry(-1.6425897326318042) q[4];
ry(0.774198714092643) q[5];
cx q[4],q[5];
ry(-1.0014095627230528) q[6];
ry(2.01649394923653) q[7];
cx q[6],q[7];
ry(2.8920730199282674) q[6];
ry(2.789070293529975) q[7];
cx q[6],q[7];
ry(2.3709468116483405) q[8];
ry(1.698453718992659) q[9];
cx q[8],q[9];
ry(1.5687414839312666) q[8];
ry(1.6578973194557607) q[9];
cx q[8],q[9];
ry(-2.5269408501890425) q[10];
ry(-1.4891921595499558) q[11];
cx q[10],q[11];
ry(-0.751709910260692) q[10];
ry(-1.7392966906847027) q[11];
cx q[10],q[11];
ry(-2.149037122427541) q[0];
ry(-1.4892052412495473) q[2];
cx q[0],q[2];
ry(-1.880847244120351) q[0];
ry(0.6255183798644532) q[2];
cx q[0],q[2];
ry(-0.900529609016286) q[2];
ry(2.0887249324515933) q[4];
cx q[2],q[4];
ry(-1.5731875362052647) q[2];
ry(-2.063582710348429) q[4];
cx q[2],q[4];
ry(-1.867010993586048) q[4];
ry(2.1381206451193613) q[6];
cx q[4],q[6];
ry(2.801577949173386) q[4];
ry(0.6339298638953172) q[6];
cx q[4],q[6];
ry(2.8001370418489704) q[6];
ry(-0.6396943473559338) q[8];
cx q[6],q[8];
ry(1.2166398202285063) q[6];
ry(1.2397680642877382) q[8];
cx q[6],q[8];
ry(0.5870513874805307) q[8];
ry(-2.9454912724271294) q[10];
cx q[8],q[10];
ry(0.732223560746581) q[8];
ry(-0.26763795378656086) q[10];
cx q[8],q[10];
ry(3.0063882270867945) q[1];
ry(-1.4678044114758713) q[3];
cx q[1],q[3];
ry(1.4791076285198628) q[1];
ry(-1.5343432735029243) q[3];
cx q[1],q[3];
ry(2.053987198736822) q[3];
ry(1.8163084989181622) q[5];
cx q[3],q[5];
ry(1.5057021380763749) q[3];
ry(2.1472736413155253) q[5];
cx q[3],q[5];
ry(0.7488385365498996) q[5];
ry(-2.3181672977791883) q[7];
cx q[5],q[7];
ry(-2.259244885559789) q[5];
ry(1.7991332246586402) q[7];
cx q[5],q[7];
ry(0.48940121611209286) q[7];
ry(0.2644104456905598) q[9];
cx q[7],q[9];
ry(-1.0910602930258615) q[7];
ry(-2.2096527140343705) q[9];
cx q[7],q[9];
ry(-0.8237919323901265) q[9];
ry(-0.6428012046715238) q[11];
cx q[9],q[11];
ry(-1.9936083743233706) q[9];
ry(-0.6435160856515999) q[11];
cx q[9],q[11];
ry(2.4605702366583886) q[0];
ry(-0.11728888846760402) q[1];
cx q[0],q[1];
ry(0.22694555118733284) q[0];
ry(-0.3551711629370574) q[1];
cx q[0],q[1];
ry(-1.7720146713985812) q[2];
ry(2.9500634142796796) q[3];
cx q[2],q[3];
ry(2.6722182579877387) q[2];
ry(1.96094709429775) q[3];
cx q[2],q[3];
ry(-1.8424283872770015) q[4];
ry(-0.8813838848982174) q[5];
cx q[4],q[5];
ry(0.3839216983730388) q[4];
ry(-3.0646812780819226) q[5];
cx q[4],q[5];
ry(0.9775836228479701) q[6];
ry(-1.0087261959410538) q[7];
cx q[6],q[7];
ry(1.9894623354245242) q[6];
ry(0.8395211077839926) q[7];
cx q[6],q[7];
ry(-0.947381650478537) q[8];
ry(-1.1494601234835402) q[9];
cx q[8],q[9];
ry(-2.4931667356858522) q[8];
ry(-2.8767776893492325) q[9];
cx q[8],q[9];
ry(-2.974488313592925) q[10];
ry(2.9730048954689754) q[11];
cx q[10],q[11];
ry(0.48079743733748875) q[10];
ry(-1.1832257863463382) q[11];
cx q[10],q[11];
ry(1.434347479433043) q[0];
ry(1.5531499398742166) q[2];
cx q[0],q[2];
ry(-1.3139458578160579) q[0];
ry(0.948350485117755) q[2];
cx q[0],q[2];
ry(2.0653063589061427) q[2];
ry(1.7210348936975777) q[4];
cx q[2],q[4];
ry(2.850550293558004) q[2];
ry(1.2908092201520944) q[4];
cx q[2],q[4];
ry(-0.35546166669336665) q[4];
ry(2.2184496163335705) q[6];
cx q[4],q[6];
ry(-0.8320473528853691) q[4];
ry(2.0022433966630864) q[6];
cx q[4],q[6];
ry(2.8442383818947525) q[6];
ry(2.0665974527327213) q[8];
cx q[6],q[8];
ry(-1.2810627915148045) q[6];
ry(0.6874568171936951) q[8];
cx q[6],q[8];
ry(0.9521318700294412) q[8];
ry(1.7598136373277808) q[10];
cx q[8],q[10];
ry(1.9900541661201316) q[8];
ry(0.46310848229258283) q[10];
cx q[8],q[10];
ry(0.321749806466011) q[1];
ry(1.669041232909103) q[3];
cx q[1],q[3];
ry(-2.923633165184808) q[1];
ry(-1.8055232121310818) q[3];
cx q[1],q[3];
ry(1.5620577155515454) q[3];
ry(2.463440544265975) q[5];
cx q[3],q[5];
ry(1.4946618147519644) q[3];
ry(-0.23379708349053097) q[5];
cx q[3],q[5];
ry(-0.5218753263808011) q[5];
ry(-1.9885893454822814) q[7];
cx q[5],q[7];
ry(1.5496682454336472) q[5];
ry(-1.623018109878199) q[7];
cx q[5],q[7];
ry(-2.1016163940422983) q[7];
ry(2.9879491416536528) q[9];
cx q[7],q[9];
ry(1.1381411687005372) q[7];
ry(-0.5265503304492556) q[9];
cx q[7],q[9];
ry(1.4467661650188557) q[9];
ry(2.439617509293816) q[11];
cx q[9],q[11];
ry(-3.004083723353921) q[9];
ry(-2.648149730058613) q[11];
cx q[9],q[11];
ry(1.708867527820768) q[0];
ry(-1.07330109405976) q[1];
cx q[0],q[1];
ry(-2.74128811608322) q[0];
ry(1.7432389887986413) q[1];
cx q[0],q[1];
ry(-2.296290946609103) q[2];
ry(2.6911762642398704) q[3];
cx q[2],q[3];
ry(-0.7075933735867734) q[2];
ry(1.1103342554783526) q[3];
cx q[2],q[3];
ry(0.22241400004305378) q[4];
ry(-1.5478290177573015) q[5];
cx q[4],q[5];
ry(-0.8811809513368367) q[4];
ry(-0.9856525431108505) q[5];
cx q[4],q[5];
ry(-2.3154884926575128) q[6];
ry(0.7525074626805894) q[7];
cx q[6],q[7];
ry(-1.5973378053814953) q[6];
ry(1.9641429459771762) q[7];
cx q[6],q[7];
ry(1.2006032724643225) q[8];
ry(-1.9514040165229065) q[9];
cx q[8],q[9];
ry(1.728239333218273) q[8];
ry(1.3643806659631978) q[9];
cx q[8],q[9];
ry(2.103927103878742) q[10];
ry(1.716103386217751) q[11];
cx q[10],q[11];
ry(-2.527916709484797) q[10];
ry(1.3293835141402262) q[11];
cx q[10],q[11];
ry(-1.9095545822271465) q[0];
ry(-1.0328294494856414) q[2];
cx q[0],q[2];
ry(-0.6574728078300462) q[0];
ry(-0.4043998930404689) q[2];
cx q[0],q[2];
ry(1.7110764315443472) q[2];
ry(0.6512770926167092) q[4];
cx q[2],q[4];
ry(2.253196354403688) q[2];
ry(-2.251097296328571) q[4];
cx q[2],q[4];
ry(2.655919914023475) q[4];
ry(1.3085470305630977) q[6];
cx q[4],q[6];
ry(1.3528272370819054) q[4];
ry(1.651814443707031) q[6];
cx q[4],q[6];
ry(-1.9237390262027114) q[6];
ry(-2.214230592916915) q[8];
cx q[6],q[8];
ry(3.1255356541938006) q[6];
ry(-1.4804092653172558) q[8];
cx q[6],q[8];
ry(-1.9801074670735264) q[8];
ry(2.6593600988716535) q[10];
cx q[8],q[10];
ry(1.0558426561940832) q[8];
ry(2.755990166312436) q[10];
cx q[8],q[10];
ry(0.5722827150946912) q[1];
ry(-2.706709742919297) q[3];
cx q[1],q[3];
ry(0.20337445055841652) q[1];
ry(-1.2256158312036467) q[3];
cx q[1],q[3];
ry(1.966131643669124) q[3];
ry(-2.512976864395068) q[5];
cx q[3],q[5];
ry(2.8173922092448236) q[3];
ry(2.2232028811355717) q[5];
cx q[3],q[5];
ry(-2.5091545419761756) q[5];
ry(2.609864761945231) q[7];
cx q[5],q[7];
ry(2.549992319716861) q[5];
ry(2.4518487193100285) q[7];
cx q[5],q[7];
ry(1.388208643730082) q[7];
ry(1.5233512761544032) q[9];
cx q[7],q[9];
ry(-1.1001100149792222) q[7];
ry(-2.4505649324637035) q[9];
cx q[7],q[9];
ry(-2.6629138137689785) q[9];
ry(1.767645175056911) q[11];
cx q[9],q[11];
ry(-1.3509075150615484) q[9];
ry(-1.8484976954892203) q[11];
cx q[9],q[11];
ry(-2.8308654211119357) q[0];
ry(-3.010614041118161) q[1];
cx q[0],q[1];
ry(-2.4474717260059515) q[0];
ry(-1.6282001370482702) q[1];
cx q[0],q[1];
ry(-1.491160363931184) q[2];
ry(-1.9950174991192038) q[3];
cx q[2],q[3];
ry(-0.6835695596839887) q[2];
ry(-1.740364558281183) q[3];
cx q[2],q[3];
ry(1.2478300480988809) q[4];
ry(-0.10559229859677582) q[5];
cx q[4],q[5];
ry(-2.4712831216521285) q[4];
ry(1.2009630310843376) q[5];
cx q[4],q[5];
ry(2.69216750306498) q[6];
ry(-0.6218302451553841) q[7];
cx q[6],q[7];
ry(-0.16734591580488978) q[6];
ry(-1.4358343119117738) q[7];
cx q[6],q[7];
ry(-0.047999463466096515) q[8];
ry(-0.3896444961074019) q[9];
cx q[8],q[9];
ry(-0.38267053336063583) q[8];
ry(-1.1472982486473142) q[9];
cx q[8],q[9];
ry(1.6035869362237591) q[10];
ry(-0.8365620349185009) q[11];
cx q[10],q[11];
ry(-1.2803804874271505) q[10];
ry(-2.169530194758784) q[11];
cx q[10],q[11];
ry(-2.118041678659527) q[0];
ry(-1.544812637330046) q[2];
cx q[0],q[2];
ry(-2.3358263749938417) q[0];
ry(-2.08519450456725) q[2];
cx q[0],q[2];
ry(2.5550950720219485) q[2];
ry(-1.1778324661654898) q[4];
cx q[2],q[4];
ry(2.0338377629296227) q[2];
ry(-2.7147333176347277) q[4];
cx q[2],q[4];
ry(-2.630163381995961) q[4];
ry(-2.913827783049624) q[6];
cx q[4],q[6];
ry(-2.1882187204701684) q[4];
ry(-2.1274021717026077) q[6];
cx q[4],q[6];
ry(2.3075596063808517) q[6];
ry(0.08951219507365123) q[8];
cx q[6],q[8];
ry(2.185641312490516) q[6];
ry(0.3906216909122566) q[8];
cx q[6],q[8];
ry(1.2810054052619226) q[8];
ry(1.3305396756016483) q[10];
cx q[8],q[10];
ry(-0.8205545530251356) q[8];
ry(2.011904467388254) q[10];
cx q[8],q[10];
ry(1.4589214381594333) q[1];
ry(1.1579059448710725) q[3];
cx q[1],q[3];
ry(0.758130278593292) q[1];
ry(0.2466427581334042) q[3];
cx q[1],q[3];
ry(2.545947488716371) q[3];
ry(2.510969809308733) q[5];
cx q[3],q[5];
ry(1.7909230371483147) q[3];
ry(-0.513903976252687) q[5];
cx q[3],q[5];
ry(-3.0274338727101706) q[5];
ry(1.4195634583507142) q[7];
cx q[5],q[7];
ry(-0.7985935135183979) q[5];
ry(1.3981512624412868) q[7];
cx q[5],q[7];
ry(2.451538020482289) q[7];
ry(-0.44374302458841075) q[9];
cx q[7],q[9];
ry(0.9379807471380737) q[7];
ry(-2.0534028692041733) q[9];
cx q[7],q[9];
ry(1.456034328112384) q[9];
ry(-2.333117891775504) q[11];
cx q[9],q[11];
ry(-1.1822794895122826) q[9];
ry(1.8027267980438653) q[11];
cx q[9],q[11];
ry(-2.730570995378836) q[0];
ry(0.8035156353203724) q[1];
cx q[0],q[1];
ry(2.513052923599648) q[0];
ry(-0.34103037022526894) q[1];
cx q[0],q[1];
ry(2.682338442597916) q[2];
ry(2.9524142399471116) q[3];
cx q[2],q[3];
ry(-0.8954419380291698) q[2];
ry(2.2233820319229993) q[3];
cx q[2],q[3];
ry(2.1403297656521385) q[4];
ry(-1.8828938774448116) q[5];
cx q[4],q[5];
ry(0.6042489228426993) q[4];
ry(-0.5898851715578926) q[5];
cx q[4],q[5];
ry(-2.5472418061486004) q[6];
ry(0.5687889318950772) q[7];
cx q[6],q[7];
ry(1.1761799967837268) q[6];
ry(-2.6042996770931577) q[7];
cx q[6],q[7];
ry(0.935774031628201) q[8];
ry(-2.6664594146559515) q[9];
cx q[8],q[9];
ry(-2.385909378320338) q[8];
ry(2.799850836533054) q[9];
cx q[8],q[9];
ry(-0.8436913718770275) q[10];
ry(1.399733969461621) q[11];
cx q[10],q[11];
ry(-3.115114288837673) q[10];
ry(-2.1540968851222315) q[11];
cx q[10],q[11];
ry(0.34983924444169534) q[0];
ry(-0.5780489773288311) q[2];
cx q[0],q[2];
ry(-2.915377841537541) q[0];
ry(1.2505806993783202) q[2];
cx q[0],q[2];
ry(-1.8279910644750255) q[2];
ry(-1.42568029106945) q[4];
cx q[2],q[4];
ry(-2.3841639737966447) q[2];
ry(-2.5629250393282845) q[4];
cx q[2],q[4];
ry(1.4066006260526338) q[4];
ry(1.7537357977872858) q[6];
cx q[4],q[6];
ry(-1.674703219178368) q[4];
ry(-0.5379632154339534) q[6];
cx q[4],q[6];
ry(2.2957345414951584) q[6];
ry(1.4994427876580914) q[8];
cx q[6],q[8];
ry(1.7463900638900647) q[6];
ry(-2.0650793455103793) q[8];
cx q[6],q[8];
ry(-1.803167186253168) q[8];
ry(-0.5539759963465879) q[10];
cx q[8],q[10];
ry(0.6849309937019655) q[8];
ry(2.599727898275002) q[10];
cx q[8],q[10];
ry(2.7642264503424707) q[1];
ry(-0.11577031925627221) q[3];
cx q[1],q[3];
ry(3.014480196856861) q[1];
ry(0.15825957660858503) q[3];
cx q[1],q[3];
ry(-0.6086975658032143) q[3];
ry(1.4538448207370995) q[5];
cx q[3],q[5];
ry(-1.5573682302117957) q[3];
ry(-2.1717587038092736) q[5];
cx q[3],q[5];
ry(-1.3260658360286748) q[5];
ry(2.6387166848771275) q[7];
cx q[5],q[7];
ry(2.4374052578882957) q[5];
ry(0.20621561287190543) q[7];
cx q[5],q[7];
ry(-1.4216077245817482) q[7];
ry(1.3039440193462344) q[9];
cx q[7],q[9];
ry(-1.0253713222607992) q[7];
ry(-1.2527491318351907) q[9];
cx q[7],q[9];
ry(0.2699348640228641) q[9];
ry(0.5853839142415467) q[11];
cx q[9],q[11];
ry(1.148597743217623) q[9];
ry(2.3091638751622527) q[11];
cx q[9],q[11];
ry(2.122985972741673) q[0];
ry(1.3110617827517341) q[1];
cx q[0],q[1];
ry(1.278594161647975) q[0];
ry(0.5142889424917954) q[1];
cx q[0],q[1];
ry(0.4141289450814458) q[2];
ry(2.104517645408852) q[3];
cx q[2],q[3];
ry(2.325077933517422) q[2];
ry(2.894779199820051) q[3];
cx q[2],q[3];
ry(-2.248330906543993) q[4];
ry(2.2611491835161575) q[5];
cx q[4],q[5];
ry(-0.7051706405022449) q[4];
ry(0.8288798740959611) q[5];
cx q[4],q[5];
ry(-1.9314653692374124) q[6];
ry(-2.500181153020958) q[7];
cx q[6],q[7];
ry(0.5915925621150299) q[6];
ry(0.8698088312407899) q[7];
cx q[6],q[7];
ry(1.098561818455746) q[8];
ry(-3.0885435983091463) q[9];
cx q[8],q[9];
ry(-0.8136183538541212) q[8];
ry(2.8297732875596795) q[9];
cx q[8],q[9];
ry(-1.9873374583390764) q[10];
ry(1.8893329271296648) q[11];
cx q[10],q[11];
ry(-0.14903466774624352) q[10];
ry(-2.5624662641477065) q[11];
cx q[10],q[11];
ry(-2.9788199385982757) q[0];
ry(-1.5364292548247247) q[2];
cx q[0],q[2];
ry(2.959186884712475) q[0];
ry(-1.9229772273317165) q[2];
cx q[0],q[2];
ry(2.9019055589356273) q[2];
ry(1.1484876141322378) q[4];
cx q[2],q[4];
ry(2.643967014113618) q[2];
ry(-1.2604844690274506) q[4];
cx q[2],q[4];
ry(-1.0188181293879026) q[4];
ry(2.3051614238008584) q[6];
cx q[4],q[6];
ry(-1.8565013774926657) q[4];
ry(-2.5775058734944056) q[6];
cx q[4],q[6];
ry(-0.5867944051530705) q[6];
ry(0.9413496715786556) q[8];
cx q[6],q[8];
ry(-0.8612719756525783) q[6];
ry(-1.3178683987419797) q[8];
cx q[6],q[8];
ry(2.7250641100354698) q[8];
ry(1.1777635084295772) q[10];
cx q[8],q[10];
ry(-2.318739547948335) q[8];
ry(1.2350118961626375) q[10];
cx q[8],q[10];
ry(2.58855829165057) q[1];
ry(-0.25617546342918107) q[3];
cx q[1],q[3];
ry(-1.0705410730691813) q[1];
ry(-2.2581505607807584) q[3];
cx q[1],q[3];
ry(1.6913437472142672) q[3];
ry(1.635119358752535) q[5];
cx q[3],q[5];
ry(1.3664746776964165) q[3];
ry(-0.15280753497061936) q[5];
cx q[3],q[5];
ry(1.5422212916568967) q[5];
ry(-1.414145026642088) q[7];
cx q[5],q[7];
ry(2.4275505727779665) q[5];
ry(-0.5842080579144519) q[7];
cx q[5],q[7];
ry(-2.077493225298642) q[7];
ry(1.3535268269427603) q[9];
cx q[7],q[9];
ry(2.4804345242097927) q[7];
ry(-1.7260888641655565) q[9];
cx q[7],q[9];
ry(-1.1817326265417751) q[9];
ry(-0.6059200604249402) q[11];
cx q[9],q[11];
ry(2.240314384162226) q[9];
ry(-2.311721992248528) q[11];
cx q[9],q[11];
ry(0.37518776347373317) q[0];
ry(-3.139631419034074) q[1];
cx q[0],q[1];
ry(-0.9054993255607071) q[0];
ry(0.6654645255496621) q[1];
cx q[0],q[1];
ry(-0.3497872046443282) q[2];
ry(2.221776975013213) q[3];
cx q[2],q[3];
ry(-2.875536583546446) q[2];
ry(2.564194478173293) q[3];
cx q[2],q[3];
ry(-2.7700720050996916) q[4];
ry(0.5945523605149476) q[5];
cx q[4],q[5];
ry(-2.7441345956918375) q[4];
ry(-2.952715764376119) q[5];
cx q[4],q[5];
ry(-0.8265938051459658) q[6];
ry(0.7492835932225652) q[7];
cx q[6],q[7];
ry(2.4248658402484327) q[6];
ry(2.638042947656212) q[7];
cx q[6],q[7];
ry(3.1024534599910907) q[8];
ry(0.8233149360817844) q[9];
cx q[8],q[9];
ry(0.3293331197310549) q[8];
ry(-1.1882220425844279) q[9];
cx q[8],q[9];
ry(-1.3190839279498254) q[10];
ry(-0.9373955192658131) q[11];
cx q[10],q[11];
ry(-0.689724999292519) q[10];
ry(-2.9872910501826184) q[11];
cx q[10],q[11];
ry(2.4648244452481496) q[0];
ry(-0.27646482140678597) q[2];
cx q[0],q[2];
ry(-0.6584774878078921) q[0];
ry(0.48740741010371236) q[2];
cx q[0],q[2];
ry(2.316497221738703) q[2];
ry(2.1047669959514206) q[4];
cx q[2],q[4];
ry(2.451087460108162) q[2];
ry(-1.2547334614563808) q[4];
cx q[2],q[4];
ry(-3.1099066487748557) q[4];
ry(-2.9204553391225794) q[6];
cx q[4],q[6];
ry(1.0682032480289667) q[4];
ry(-0.4381453778909016) q[6];
cx q[4],q[6];
ry(3.054506498005705) q[6];
ry(-2.7501225361404384) q[8];
cx q[6],q[8];
ry(-1.6335028332047734) q[6];
ry(1.1796786257505716) q[8];
cx q[6],q[8];
ry(2.169284538947239) q[8];
ry(2.2089623105355196) q[10];
cx q[8],q[10];
ry(-0.5598492130038982) q[8];
ry(2.763493899037165) q[10];
cx q[8],q[10];
ry(-0.9498094392626727) q[1];
ry(2.662467093836344) q[3];
cx q[1],q[3];
ry(1.6688478567006007) q[1];
ry(-2.158567528791463) q[3];
cx q[1],q[3];
ry(-1.4687107768779422) q[3];
ry(-2.9284758603634) q[5];
cx q[3],q[5];
ry(2.9503103148023384) q[3];
ry(-2.5828731429654943) q[5];
cx q[3],q[5];
ry(-2.9353869435115714) q[5];
ry(1.5847397092277593) q[7];
cx q[5],q[7];
ry(-2.9718282339641746) q[5];
ry(-3.1280597399817625) q[7];
cx q[5],q[7];
ry(-2.068094584514829) q[7];
ry(-2.574263984373204) q[9];
cx q[7],q[9];
ry(2.8263746008771338) q[7];
ry(2.8383009419353153) q[9];
cx q[7],q[9];
ry(2.180698742552557) q[9];
ry(2.7799295785637055) q[11];
cx q[9],q[11];
ry(0.7447599448911698) q[9];
ry(3.09766939644539) q[11];
cx q[9],q[11];
ry(-2.4820922843124618) q[0];
ry(0.3551830594734984) q[1];
cx q[0],q[1];
ry(-1.1350432878999426) q[0];
ry(1.235123440653398) q[1];
cx q[0],q[1];
ry(0.9532304623461956) q[2];
ry(0.22717884163409235) q[3];
cx q[2],q[3];
ry(-1.4979764788831496) q[2];
ry(-2.194566107250723) q[3];
cx q[2],q[3];
ry(-0.4318737986530268) q[4];
ry(-0.5573434851561885) q[5];
cx q[4],q[5];
ry(-1.1322628584343102) q[4];
ry(2.3647494869814856) q[5];
cx q[4],q[5];
ry(-1.9432503583668035) q[6];
ry(1.397705928317029) q[7];
cx q[6],q[7];
ry(-1.8303425092032601) q[6];
ry(-1.0236125062037256) q[7];
cx q[6],q[7];
ry(2.7338000916042917) q[8];
ry(0.5627763238675136) q[9];
cx q[8],q[9];
ry(2.1039193246412773) q[8];
ry(-0.6366559134639891) q[9];
cx q[8],q[9];
ry(1.9956037112117073) q[10];
ry(-3.010050728683616) q[11];
cx q[10],q[11];
ry(1.551581957486782) q[10];
ry(0.49468105197266915) q[11];
cx q[10],q[11];
ry(-2.278636437576063) q[0];
ry(1.767146496157225) q[2];
cx q[0],q[2];
ry(0.5564534487823618) q[0];
ry(-0.671684019341515) q[2];
cx q[0],q[2];
ry(-0.860518436160822) q[2];
ry(-2.7862444821175156) q[4];
cx q[2],q[4];
ry(2.2930499770383532) q[2];
ry(-2.7135851184551534) q[4];
cx q[2],q[4];
ry(-2.362229100559484) q[4];
ry(0.7418094415375854) q[6];
cx q[4],q[6];
ry(-0.7665007808155622) q[4];
ry(-2.770188922084104) q[6];
cx q[4],q[6];
ry(-2.938431263713752) q[6];
ry(-1.1317701694299076) q[8];
cx q[6],q[8];
ry(1.7078704490260512) q[6];
ry(0.991337295419422) q[8];
cx q[6],q[8];
ry(0.5075756842802086) q[8];
ry(1.3103204054995392) q[10];
cx q[8],q[10];
ry(-2.517385109617995) q[8];
ry(-0.5453681430051835) q[10];
cx q[8],q[10];
ry(2.1092601947789236) q[1];
ry(2.058594344791083) q[3];
cx q[1],q[3];
ry(-2.3204983640251133) q[1];
ry(-2.131949713968701) q[3];
cx q[1],q[3];
ry(-2.5811972105247905) q[3];
ry(2.658119288148655) q[5];
cx q[3],q[5];
ry(-1.451375387721942) q[3];
ry(-1.607294802729493) q[5];
cx q[3],q[5];
ry(0.13369299566978896) q[5];
ry(0.2049560184279662) q[7];
cx q[5],q[7];
ry(-0.9442035209652262) q[5];
ry(1.6686991527252666) q[7];
cx q[5],q[7];
ry(2.163276800375894) q[7];
ry(1.8590622860189472) q[9];
cx q[7],q[9];
ry(-0.7851875326506438) q[7];
ry(-0.6507590170822922) q[9];
cx q[7],q[9];
ry(-1.19348803213326) q[9];
ry(-0.052846001924242714) q[11];
cx q[9],q[11];
ry(-1.0421273202866514) q[9];
ry(-0.31610986703967203) q[11];
cx q[9],q[11];
ry(-2.2944908087493365) q[0];
ry(0.8800616273369557) q[1];
cx q[0],q[1];
ry(-1.5711949101311191) q[0];
ry(1.5139465327275543) q[1];
cx q[0],q[1];
ry(0.5464505672913502) q[2];
ry(-2.6902861405072827) q[3];
cx q[2],q[3];
ry(1.5809101642917902) q[2];
ry(-2.141933838671023) q[3];
cx q[2],q[3];
ry(1.5895930187663048) q[4];
ry(1.3394087757334572) q[5];
cx q[4],q[5];
ry(-2.7039078118064372) q[4];
ry(1.769683301400441) q[5];
cx q[4],q[5];
ry(-2.983054722367731) q[6];
ry(-2.8473983912977925) q[7];
cx q[6],q[7];
ry(1.6662964888333418) q[6];
ry(1.693643774981278) q[7];
cx q[6],q[7];
ry(-0.3901724203382928) q[8];
ry(1.6361651312640302) q[9];
cx q[8],q[9];
ry(-1.8656937222948273) q[8];
ry(-2.4809907978583983) q[9];
cx q[8],q[9];
ry(2.8793804223858674) q[10];
ry(0.20269172785727285) q[11];
cx q[10],q[11];
ry(-1.8807027921134525) q[10];
ry(1.2164904972114545) q[11];
cx q[10],q[11];
ry(-0.49119556066618875) q[0];
ry(2.5241765277769934) q[2];
cx q[0],q[2];
ry(-1.595563517998903) q[0];
ry(2.4034376158360256) q[2];
cx q[0],q[2];
ry(0.7424860367364605) q[2];
ry(-3.0704568416327613) q[4];
cx q[2],q[4];
ry(-3.075391511126051) q[2];
ry(1.079667509755855) q[4];
cx q[2],q[4];
ry(1.9473068215695852) q[4];
ry(-2.8402905409092) q[6];
cx q[4],q[6];
ry(2.5777260548433047) q[4];
ry(-2.1945034917321626) q[6];
cx q[4],q[6];
ry(2.4649110769030047) q[6];
ry(-0.8816128066710305) q[8];
cx q[6],q[8];
ry(0.7435572171767725) q[6];
ry(-1.533129503677106) q[8];
cx q[6],q[8];
ry(0.6612309873995876) q[8];
ry(-0.5647888958259699) q[10];
cx q[8],q[10];
ry(2.2776824110443363) q[8];
ry(1.1669186691952533) q[10];
cx q[8],q[10];
ry(1.8033886799928442) q[1];
ry(2.4365794890361845) q[3];
cx q[1],q[3];
ry(-2.695112892837789) q[1];
ry(-2.565886042339866) q[3];
cx q[1],q[3];
ry(2.8798089233169644) q[3];
ry(1.0285654678718288) q[5];
cx q[3],q[5];
ry(0.3591228245724638) q[3];
ry(-1.0731229452818791) q[5];
cx q[3],q[5];
ry(0.8554951036875389) q[5];
ry(1.6293258284519259) q[7];
cx q[5],q[7];
ry(-2.2724460240583237) q[5];
ry(-2.1940766297362426) q[7];
cx q[5],q[7];
ry(1.7106828452247296) q[7];
ry(-2.773827140284445) q[9];
cx q[7],q[9];
ry(-0.3906954369795669) q[7];
ry(2.323018528721841) q[9];
cx q[7],q[9];
ry(-1.4423445902159848) q[9];
ry(-2.160164940200374) q[11];
cx q[9],q[11];
ry(-2.483365785056631) q[9];
ry(2.644430991190171) q[11];
cx q[9],q[11];
ry(-2.7245732888832825) q[0];
ry(-0.2488164599892182) q[1];
cx q[0],q[1];
ry(-1.9994586354908863) q[0];
ry(1.561701841849728) q[1];
cx q[0],q[1];
ry(0.47219920464732384) q[2];
ry(-1.4176045318310873) q[3];
cx q[2],q[3];
ry(2.629766930563465) q[2];
ry(-1.3201837053375345) q[3];
cx q[2],q[3];
ry(2.5953511269491787) q[4];
ry(-1.0635026902607834) q[5];
cx q[4],q[5];
ry(1.5734634909539733) q[4];
ry(2.680922607227045) q[5];
cx q[4],q[5];
ry(-1.0859214301917186) q[6];
ry(-1.1876985743549966) q[7];
cx q[6],q[7];
ry(-2.7208797705135868) q[6];
ry(-2.2336848327265963) q[7];
cx q[6],q[7];
ry(-1.6831112551924168) q[8];
ry(2.588760916363933) q[9];
cx q[8],q[9];
ry(1.0014331994255388) q[8];
ry(-1.073227969151752) q[9];
cx q[8],q[9];
ry(-3.073406939778622) q[10];
ry(0.13424664946192255) q[11];
cx q[10],q[11];
ry(2.2540518623856993) q[10];
ry(0.6140829223001053) q[11];
cx q[10],q[11];
ry(0.18128255285495892) q[0];
ry(-0.04377017054007837) q[2];
cx q[0],q[2];
ry(-0.5739015689242951) q[0];
ry(-1.8521761968680488) q[2];
cx q[0],q[2];
ry(1.5836372508300078) q[2];
ry(2.348097780574816) q[4];
cx q[2],q[4];
ry(-1.9566054556982069) q[2];
ry(2.0116708627740514) q[4];
cx q[2],q[4];
ry(0.20747853966357854) q[4];
ry(-2.589808007967634) q[6];
cx q[4],q[6];
ry(1.8225824644002335) q[4];
ry(2.4243426707823916) q[6];
cx q[4],q[6];
ry(3.0910162438879936) q[6];
ry(1.7461510124480455) q[8];
cx q[6],q[8];
ry(-0.5162843676024709) q[6];
ry(-2.338096517248239) q[8];
cx q[6],q[8];
ry(-2.5019676725805415) q[8];
ry(1.6860208991967165) q[10];
cx q[8],q[10];
ry(2.0729031543462835) q[8];
ry(2.2744962656456016) q[10];
cx q[8],q[10];
ry(-1.2566552919381762) q[1];
ry(1.8524278556284877) q[3];
cx q[1],q[3];
ry(1.6621605582278813) q[1];
ry(2.764727178648187) q[3];
cx q[1],q[3];
ry(1.8480108875073906) q[3];
ry(2.906469114319674) q[5];
cx q[3],q[5];
ry(-1.2698194528254458) q[3];
ry(0.591400212146608) q[5];
cx q[3],q[5];
ry(1.4482099771206125) q[5];
ry(1.4000901046499974) q[7];
cx q[5],q[7];
ry(-2.0729169421238507) q[5];
ry(1.1941769043499315) q[7];
cx q[5],q[7];
ry(-0.9034467509294037) q[7];
ry(-2.091055929409217) q[9];
cx q[7],q[9];
ry(-2.3016090377436136) q[7];
ry(-1.6144909090971442) q[9];
cx q[7],q[9];
ry(1.9932228876557652) q[9];
ry(-0.11850346877307862) q[11];
cx q[9],q[11];
ry(1.505810959739002) q[9];
ry(-1.3164160099473015) q[11];
cx q[9],q[11];
ry(1.4903877734537152) q[0];
ry(-1.6560035739425985) q[1];
cx q[0],q[1];
ry(1.1219486945355044) q[0];
ry(1.2607393717567899) q[1];
cx q[0],q[1];
ry(2.370526421699085) q[2];
ry(0.9385438654331032) q[3];
cx q[2],q[3];
ry(-1.905277127641808) q[2];
ry(-0.39001855559135384) q[3];
cx q[2],q[3];
ry(-0.9230450191571293) q[4];
ry(-0.6275370137284861) q[5];
cx q[4],q[5];
ry(1.6709709505692478) q[4];
ry(-1.3453570151394487) q[5];
cx q[4],q[5];
ry(-1.6464823395403412) q[6];
ry(-3.020552179711283) q[7];
cx q[6],q[7];
ry(0.6131590417919396) q[6];
ry(-2.960616018349898) q[7];
cx q[6],q[7];
ry(1.5474050678574762) q[8];
ry(0.30602227523943326) q[9];
cx q[8],q[9];
ry(1.8307570000720312) q[8];
ry(-1.4090625378433321) q[9];
cx q[8],q[9];
ry(-1.9332616107561735) q[10];
ry(-2.204912770709953) q[11];
cx q[10],q[11];
ry(-2.0949119936020786) q[10];
ry(-1.3344445837229992) q[11];
cx q[10],q[11];
ry(-2.7521234423067655) q[0];
ry(-0.8228412875677602) q[2];
cx q[0],q[2];
ry(-2.816916747998744) q[0];
ry(-0.6139798420174155) q[2];
cx q[0],q[2];
ry(-0.746389099813646) q[2];
ry(-0.29269823342145074) q[4];
cx q[2],q[4];
ry(0.4848217535420419) q[2];
ry(-2.285469733835312) q[4];
cx q[2],q[4];
ry(-2.8711475240145603) q[4];
ry(0.1868153972591191) q[6];
cx q[4],q[6];
ry(-2.159763911909681) q[4];
ry(-2.935294084649367) q[6];
cx q[4],q[6];
ry(-1.1665581210583742) q[6];
ry(0.9329502477173961) q[8];
cx q[6],q[8];
ry(-2.627389056979529) q[6];
ry(0.8159148558205329) q[8];
cx q[6],q[8];
ry(-0.5448487551999956) q[8];
ry(-0.9951038353523652) q[10];
cx q[8],q[10];
ry(-2.97547398703158) q[8];
ry(-1.1237893829061094) q[10];
cx q[8],q[10];
ry(-1.4363532264519323) q[1];
ry(-0.5949813159665203) q[3];
cx q[1],q[3];
ry(2.9087538154069112) q[1];
ry(-2.613458546584513) q[3];
cx q[1],q[3];
ry(-1.754042405809735) q[3];
ry(-0.7435046973391538) q[5];
cx q[3],q[5];
ry(2.7541453921005985) q[3];
ry(2.5565583227684163) q[5];
cx q[3],q[5];
ry(1.3580334412350048) q[5];
ry(-1.5826020593609857) q[7];
cx q[5],q[7];
ry(0.5532302938740425) q[5];
ry(-2.5106909065580174) q[7];
cx q[5],q[7];
ry(-2.434281147537281) q[7];
ry(-0.40815915966023636) q[9];
cx q[7],q[9];
ry(1.0804573568522127) q[7];
ry(1.393282897363514) q[9];
cx q[7],q[9];
ry(-2.7019615862893587) q[9];
ry(-2.8217204205206365) q[11];
cx q[9],q[11];
ry(2.0506121013817467) q[9];
ry(1.101059085580593) q[11];
cx q[9],q[11];
ry(0.06057771299982928) q[0];
ry(2.826537959569161) q[1];
ry(1.6007047897860918) q[2];
ry(-2.7495053652752204) q[3];
ry(-0.3416313517832972) q[4];
ry(-2.697461897754259) q[5];
ry(-0.4793812172416274) q[6];
ry(0.38899380327945443) q[7];
ry(1.5056577441733783) q[8];
ry(0.3628602131115787) q[9];
ry(0.1613654861158187) q[10];
ry(0.7101228087904894) q[11];