OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-3.0251593035118987) q[0];
rz(-2.5610790789589557) q[0];
ry(3.1414200126208813) q[1];
rz(0.16885054980260977) q[1];
ry(2.263625766915827) q[2];
rz(0.2265218470168051) q[2];
ry(0.007418570207907831) q[3];
rz(-1.1920753530948192) q[3];
ry(1.491313854051655) q[4];
rz(-0.19291011735587177) q[4];
ry(2.2171174007898538) q[5];
rz(-2.2196016546342907) q[5];
ry(3.1251651740532767) q[6];
rz(-1.8523019379574786) q[6];
ry(-3.141556131063392) q[7];
rz(-2.941189754362696) q[7];
ry(-2.651485083231014) q[8];
rz(1.2875580525065897) q[8];
ry(2.081244115081951) q[9];
rz(1.1532433173696233) q[9];
ry(-2.397315886135891) q[10];
rz(2.4853348502268355) q[10];
ry(-0.10226385597511012) q[11];
rz(-0.9335431141619751) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.45323464021671583) q[0];
rz(1.4166753839124437) q[0];
ry(-0.0024281744880498835) q[1];
rz(-2.455226217190624) q[1];
ry(-0.3747239089560932) q[2];
rz(-0.13974418373292563) q[2];
ry(0.9102129313498146) q[3];
rz(-1.697665152947784) q[3];
ry(-1.820802374803277) q[4];
rz(-2.3514257314212177) q[4];
ry(1.8737412178599993) q[5];
rz(3.0261116669099914) q[5];
ry(0.511547315937606) q[6];
rz(-2.4626279612317714) q[6];
ry(-1.6045079723089126) q[7];
rz(0.125682631998659) q[7];
ry(1.4085280952382961) q[8];
rz(1.819886568209341) q[8];
ry(-0.7843219292964261) q[9];
rz(1.9739079457922681) q[9];
ry(-1.4081745018896268) q[10];
rz(2.6510432565526703) q[10];
ry(2.603805532009997) q[11];
rz(-2.7343029606272435) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-3.0687357531364516) q[0];
rz(-1.2837650120310453) q[0];
ry(-0.01825366756603798) q[1];
rz(1.0028330807076244) q[1];
ry(-2.324681102969305) q[2];
rz(1.8237313423633228) q[2];
ry(2.4278535873837574) q[3];
rz(-0.018331933513383348) q[3];
ry(0.04781265342418783) q[4];
rz(-2.8966028698949753) q[4];
ry(1.8630594216497471) q[5];
rz(1.2151180757054174) q[5];
ry(3.07022857627514) q[6];
rz(1.5379763693405426) q[6];
ry(0.0007030183528069499) q[7];
rz(1.534411524293206) q[7];
ry(3.1332144864768123) q[8];
rz(-2.6717090888906116) q[8];
ry(-1.7306641563493406) q[9];
rz(-2.7274819620966517) q[9];
ry(2.9879438143558117) q[10];
rz(-1.6786971974965317) q[10];
ry(0.01391055560720922) q[11];
rz(-2.9153313613143728) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.7914556125500984) q[0];
rz(-1.0175711993858532) q[0];
ry(-0.0002799136361302246) q[1];
rz(-0.943244120975959) q[1];
ry(-3.0585411216800384) q[2];
rz(3.0451614215359624) q[2];
ry(-1.9084516205316673) q[3];
rz(-3.0489035323397764) q[3];
ry(0.09011199260102032) q[4];
rz(2.3753567180715733) q[4];
ry(2.6009212421077357) q[5];
rz(-2.9093958295407494) q[5];
ry(2.748198066785088) q[6];
rz(1.034134252176539) q[6];
ry(2.1077273207546683) q[7];
rz(-1.482073809009713) q[7];
ry(3.013621138175101) q[8];
rz(-0.2472008623481621) q[8];
ry(-2.981926095327946) q[9];
rz(-1.3841757950320392) q[9];
ry(2.0772557966471425) q[10];
rz(0.35193043840001664) q[10];
ry(-0.5529892581196757) q[11];
rz(0.211398134758773) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.695692449090261) q[0];
rz(0.14470474862345825) q[0];
ry(-3.099774571724094) q[1];
rz(0.6146903163404501) q[1];
ry(-2.5160288149501446) q[2];
rz(2.550351954567589) q[2];
ry(2.826376282725505) q[3];
rz(-1.8801846632034938) q[3];
ry(3.135766509996063) q[4];
rz(2.586835832133668) q[4];
ry(1.8518693393698855) q[5];
rz(-0.08662324303922553) q[5];
ry(0.033220030390212145) q[6];
rz(-2.1192291264457612) q[6];
ry(-3.139855881503825) q[7];
rz(2.8719736467895562) q[7];
ry(-0.010125641601864466) q[8];
rz(-0.9706879903397034) q[8];
ry(0.7757816047500654) q[9];
rz(1.0412589785118302) q[9];
ry(-2.489922901360372) q[10];
rz(-2.2081643233427446) q[10];
ry(-0.27551380916226387) q[11];
rz(-2.861212728001827) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.11839046501712792) q[0];
rz(0.535753716076765) q[0];
ry(2.527876274197062) q[1];
rz(0.2708084035219986) q[1];
ry(3.0466359473698246) q[2];
rz(2.7040223607361216) q[2];
ry(1.4145223310889563) q[3];
rz(-2.2794641494548267) q[3];
ry(0.029431405036500102) q[4];
rz(0.6005008282197362) q[4];
ry(-2.631232215185772) q[5];
rz(-0.4356916335638692) q[5];
ry(0.34055582002587226) q[6];
rz(0.26867859306083725) q[6];
ry(2.9487625483736384) q[7];
rz(-0.6541682906801) q[7];
ry(-2.4658001592578134) q[8];
rz(-0.7548464823343952) q[8];
ry(-2.3449436106410415) q[9];
rz(-0.3632571558330964) q[9];
ry(1.658087196623887) q[10];
rz(2.7161091470215086) q[10];
ry(3.081661384151487) q[11];
rz(-1.284629544556988) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.657725086499536) q[0];
rz(-3.058011366552235) q[0];
ry(2.0002907405698016) q[1];
rz(-0.48585804166900814) q[1];
ry(3.138767163068211) q[2];
rz(-1.5281598533042586) q[2];
ry(2.4377905485513396) q[3];
rz(-1.80937315619955) q[3];
ry(1.3632038396395956) q[4];
rz(0.7083677948203143) q[4];
ry(1.3477536980474611) q[5];
rz(-1.1299493481051899) q[5];
ry(0.06023406819780597) q[6];
rz(-0.4506143013489851) q[6];
ry(3.139415021399525) q[7];
rz(1.0104020363710138) q[7];
ry(0.006212606490255013) q[8];
rz(1.6845171882961603) q[8];
ry(1.0386522475723539) q[9];
rz(2.215555454078138) q[9];
ry(2.7644463223699276) q[10];
rz(-2.881068884769046) q[10];
ry(-0.0621299927409181) q[11];
rz(0.44690015574004516) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.8862929619140623) q[0];
rz(-1.269143789751599) q[0];
ry(-2.698439146314668) q[1];
rz(0.01670805031207545) q[1];
ry(3.141230143212815) q[2];
rz(2.356037597147802) q[2];
ry(0.9441930338821847) q[3];
rz(2.1830091603764012) q[3];
ry(3.1034590701301075) q[4];
rz(-2.8077276605922052) q[4];
ry(1.2955468443330103) q[5];
rz(-2.9872619532095004) q[5];
ry(0.3418928914337753) q[6];
rz(0.14073670651826656) q[6];
ry(-0.5416134908820309) q[7];
rz(2.9428783163839647) q[7];
ry(-1.9927719696850428) q[8];
rz(-2.387369629473878) q[8];
ry(1.2918493812288432) q[9];
rz(-0.1388654302300327) q[9];
ry(0.1739183693580246) q[10];
rz(-3.1319307280091695) q[10];
ry(0.2325930362663318) q[11];
rz(-1.2396519267040365) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.014861226601849) q[0];
rz(-3.07709844256698) q[0];
ry(-0.8123800140262633) q[1];
rz(1.7317710981488759) q[1];
ry(-0.0035836934701220582) q[2];
rz(1.1311892229890999) q[2];
ry(1.4107458817728684) q[3];
rz(-2.061598157687508) q[3];
ry(0.14451769175299667) q[4];
rz(2.3950845571517316) q[4];
ry(2.532466838385692) q[5];
rz(-2.8000955033900814) q[5];
ry(-1.7262794288864383) q[6];
rz(0.06505526447408254) q[6];
ry(0.010522029207562107) q[7];
rz(2.1640266739203486) q[7];
ry(1.42476611172963) q[8];
rz(-2.305743894354022) q[8];
ry(1.0893626178983693) q[9];
rz(0.9233371481439899) q[9];
ry(1.2803179001096883) q[10];
rz(-0.9440226873464583) q[10];
ry(2.9024354915940473) q[11];
rz(1.2908464919431966) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.16558722360149705) q[0];
rz(0.6440837400125972) q[0];
ry(0.2073832869703473) q[1];
rz(0.5365099053232818) q[1];
ry(0.0019736662648845993) q[2];
rz(-0.9082394462890475) q[2];
ry(1.1509103562922478) q[3];
rz(1.121369270113088) q[3];
ry(-0.7940477114973861) q[4];
rz(-0.6671901311546495) q[4];
ry(2.922633736440687) q[5];
rz(1.7394697914965678) q[5];
ry(1.2661321088171542) q[6];
rz(-3.1163281573021755) q[6];
ry(0.000522741340402747) q[7];
rz(1.359746290504677) q[7];
ry(0.1112429488276172) q[8];
rz(-0.8238941647971719) q[8];
ry(-0.0053416368495316675) q[9];
rz(0.3250265427218224) q[9];
ry(2.0941773705205664) q[10];
rz(-1.681674637003674) q[10];
ry(-0.5211645063970529) q[11];
rz(2.713574640509643) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.1177800020485464) q[0];
rz(-1.707267282067696) q[0];
ry(2.8147617433811605) q[1];
rz(-2.3592891402441913) q[1];
ry(-0.008959413698715224) q[2];
rz(-1.345309019565373) q[2];
ry(2.9448733186466423) q[3];
rz(-3.0106947586508728) q[3];
ry(0.3907466528573778) q[4];
rz(-2.7434696071424227) q[4];
ry(3.1240501870023527) q[5];
rz(1.5875946848502527) q[5];
ry(-1.4724317744168447) q[6];
rz(-0.2921685366467521) q[6];
ry(-3.135746715864734) q[7];
rz(-2.306387638981904) q[7];
ry(1.4737939920201493) q[8];
rz(-1.5997331057340807) q[8];
ry(0.8715001410312618) q[9];
rz(2.8673072183826807) q[9];
ry(0.7380251234409343) q[10];
rz(0.8732839374618847) q[10];
ry(0.4188896635693631) q[11];
rz(1.9805235523181441) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.5854018170229857) q[0];
rz(1.5807491515468755) q[0];
ry(-0.20440489426416592) q[1];
rz(-2.3014373437883835) q[1];
ry(0.002982195415922126) q[2];
rz(-0.8895886424184121) q[2];
ry(1.7693566240705392) q[3];
rz(2.6454726064241796) q[3];
ry(3.065054781107587) q[4];
rz(1.390416066187685) q[4];
ry(-2.3013590634273418) q[5];
rz(-2.91799003335212) q[5];
ry(-1.1761283866030687) q[6];
rz(2.188612322833624) q[6];
ry(2.081433578998901) q[7];
rz(0.6171572955646201) q[7];
ry(-3.080130763322272) q[8];
rz(-3.0554397661391572) q[8];
ry(-2.190187146716559) q[9];
rz(-3.0701837807815764) q[9];
ry(2.0026780089244154) q[10];
rz(-2.66489883139917) q[10];
ry(-0.8019478523335655) q[11];
rz(1.4480438089411096) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.8749366593749874) q[0];
rz(-2.84019029735039) q[0];
ry(-0.3585527978577937) q[1];
rz(0.7028686289163427) q[1];
ry(-3.10124301135469) q[2];
rz(1.7537921084956878) q[2];
ry(1.950914492448864) q[3];
rz(-0.005291442610850875) q[3];
ry(0.2904567311271089) q[4];
rz(2.5956439734043815) q[4];
ry(2.560244313004774) q[5];
rz(-0.5320208459930269) q[5];
ry(0.48799634179124196) q[6];
rz(-0.18765578342053715) q[6];
ry(-0.01905477466823502) q[7];
rz(-1.0140097775033992) q[7];
ry(-3.139816134035988) q[8];
rz(-1.226420538279457) q[8];
ry(-2.0448037067411136) q[9];
rz(2.7391856158374113) q[9];
ry(2.845874298612219) q[10];
rz(2.3752187024425058) q[10];
ry(3.0095524602888215) q[11];
rz(0.5458232658342812) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.657945889485814) q[0];
rz(2.717896991924881) q[0];
ry(-2.7561762812005743) q[1];
rz(-2.5390620236886483) q[1];
ry(-0.007591674398538474) q[2];
rz(-2.6114147686966365) q[2];
ry(2.584057931481293) q[3];
rz(1.9725644714050778) q[3];
ry(3.119312733347235) q[4];
rz(0.1428261386652823) q[4];
ry(3.1153529413988403) q[5];
rz(2.8078756139386796) q[5];
ry(1.506250076505767) q[6];
rz(-0.16253132914771307) q[6];
ry(-1.676507209907634) q[7];
rz(-2.5482249789650364) q[7];
ry(3.061258184693726) q[8];
rz(-1.598877034501835) q[8];
ry(2.0532095918658273) q[9];
rz(2.667614972631003) q[9];
ry(-2.6464356720969704) q[10];
rz(-2.172474424205051) q[10];
ry(2.2641829488848657) q[11];
rz(2.44216534816501) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.976995883538316) q[0];
rz(1.7509014286100633) q[0];
ry(0.1158980841097626) q[1];
rz(1.7533196885175268) q[1];
ry(-0.024997117651872576) q[2];
rz(1.9729400772259145) q[2];
ry(2.2922952383386153) q[3];
rz(-0.6108390975610796) q[3];
ry(0.38719560092818534) q[4];
rz(-1.0361903871419154) q[4];
ry(-2.945908530799241) q[5];
rz(-2.190137447000069) q[5];
ry(2.999883856164493) q[6];
rz(-0.2623969699483325) q[6];
ry(-3.138611380692785) q[7];
rz(0.5556143901513702) q[7];
ry(3.1403269437849572) q[8];
rz(2.5122258962531743) q[8];
ry(-1.364554388152185) q[9];
rz(-2.9038128150983398) q[9];
ry(-1.7221244304561374) q[10];
rz(-1.8312300261367294) q[10];
ry(0.2015843427800175) q[11];
rz(1.5788527083133008) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.6176160717935506) q[0];
rz(-2.016971216762922) q[0];
ry(-2.3694684321116735) q[1];
rz(-1.5544428172364322) q[1];
ry(3.1413439341612186) q[2];
rz(-2.048716553503442) q[2];
ry(0.4506348595836166) q[3];
rz(0.6407409476096566) q[3];
ry(-3.0821349981910156) q[4];
rz(0.11748290916022608) q[4];
ry(3.127852101713103) q[5];
rz(0.5227585739805072) q[5];
ry(-1.7877963206993703) q[6];
rz(-0.22149577132282813) q[6];
ry(-1.4452360830727224) q[7];
rz(-1.0072266545294086) q[7];
ry(-2.7202645297137944) q[8];
rz(-1.6449323428818374) q[8];
ry(2.8884113636072724) q[9];
rz(-1.1618971184269458) q[9];
ry(0.9085257075663633) q[10];
rz(-3.0085588012461773) q[10];
ry(2.696126111850253) q[11];
rz(2.6316378853460742) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.7219687697636976) q[0];
rz(2.206851567755236) q[0];
ry(-1.8046093105440255) q[1];
rz(0.9300671778407549) q[1];
ry(-2.728209617507228) q[2];
rz(0.5525149307472884) q[2];
ry(-2.380916030810741) q[3];
rz(-1.4783703653523554) q[3];
ry(2.9294420938640897) q[4];
rz(-0.4257475651569625) q[4];
ry(0.7292949156807103) q[5];
rz(-1.176316243994674) q[5];
ry(-0.49809715313331804) q[6];
rz(0.4678270574966518) q[6];
ry(-0.02041863862878301) q[7];
rz(0.10602992350100315) q[7];
ry(-0.008874805153694472) q[8];
rz(-1.211812916275563) q[8];
ry(0.10197032167690631) q[9];
rz(0.07511271965069277) q[9];
ry(1.2767570159182753) q[10];
rz(-2.7761796052884153) q[10];
ry(3.1223536373525795) q[11];
rz(-2.511461092646047) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.7772640901969692) q[0];
rz(2.3960136282896176) q[0];
ry(3.130447669596724) q[1];
rz(0.3775072402164464) q[1];
ry(2.239347439733679) q[2];
rz(-3.09490950421563) q[2];
ry(0.1337082967450155) q[3];
rz(1.1972961478007687) q[3];
ry(-1.9585966315799288) q[4];
rz(-1.8914818197672747) q[4];
ry(-0.8029075682027003) q[5];
rz(1.6392950976937475) q[5];
ry(2.6947886391758704) q[6];
rz(-1.5664999791744227) q[6];
ry(2.0055036609427743) q[7];
rz(1.1337650125746348) q[7];
ry(-1.1932563809569323) q[8];
rz(1.0462155573434346) q[8];
ry(-0.34323171509328065) q[9];
rz(-1.7515860244220027) q[9];
ry(-0.9988147974877649) q[10];
rz(-1.6945980671619711) q[10];
ry(-3.1078554922632935) q[11];
rz(0.9499099838725998) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.3814069919291387) q[0];
rz(0.6936912358058901) q[0];
ry(1.9967800525420207) q[1];
rz(0.1695296921235965) q[1];
ry(-0.6441886817180299) q[2];
rz(-0.030505069159671654) q[2];
ry(3.1413071685599068) q[3];
rz(-2.1264885939073963) q[3];
ry(-1.4133788556737459) q[4];
rz(-0.18097537050222634) q[4];
ry(-0.0649190075379995) q[5];
rz(1.5641005454146253) q[5];
ry(0.6986808678926966) q[6];
rz(2.4536278084928544) q[6];
ry(0.002583273958278731) q[7];
rz(-1.7278941028819164) q[7];
ry(0.558709727353344) q[8];
rz(0.7430479764525099) q[8];
ry(-2.252268817268167) q[9];
rz(-1.0335546824808404) q[9];
ry(0.9890292143299347) q[10];
rz(0.6682628779259475) q[10];
ry(2.575079621169209) q[11];
rz(0.7033342805449497) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.1224851769861317) q[0];
rz(0.12392365516240302) q[0];
ry(-2.673679432228061) q[1];
rz(3.0816857946158343) q[1];
ry(-1.7946108047817058) q[2];
rz(-2.354486890767723) q[2];
ry(0.018796301723706228) q[3];
rz(2.1812622091848617) q[3];
ry(-1.6795852850972341) q[4];
rz(0.9316480006528618) q[4];
ry(-3.136469627858788) q[5];
rz(0.80341060658552) q[5];
ry(0.026515950881534245) q[6];
rz(-2.500747593896686) q[6];
ry(3.133278545366011) q[7];
rz(-0.1754546343841863) q[7];
ry(2.780153234195907) q[8];
rz(-2.2342266362537275) q[8];
ry(0.0760110733005126) q[9];
rz(2.2195035050894107) q[9];
ry(0.0908543457326296) q[10];
rz(2.3909268728359585) q[10];
ry(-1.9917689254774489) q[11];
rz(1.400026264455177) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.691183559764724) q[0];
rz(-2.3236443699130875) q[0];
ry(-1.031695773355354) q[1];
rz(1.0502745890000476) q[1];
ry(0.005783568154844154) q[2];
rz(0.8591579042065633) q[2];
ry(3.0926798059305973) q[3];
rz(-0.4808159039315311) q[3];
ry(2.7087778165237717) q[4];
rz(-1.1767571316706102) q[4];
ry(0.017168269914769496) q[5];
rz(-2.9336357547322063) q[5];
ry(-0.7411465670421036) q[6];
rz(-1.2230941328641818) q[6];
ry(-1.9084313974185179) q[7];
rz(-2.6237522141430274) q[7];
ry(-1.8410030371162842) q[8];
rz(2.8478310949748096) q[8];
ry(-2.6119153474177383) q[9];
rz(-0.6520207961690879) q[9];
ry(0.08267158852223044) q[10];
rz(1.6908190920506376) q[10];
ry(0.6264943527930997) q[11];
rz(-2.0718077803203316) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.05312685831867194) q[0];
rz(-1.3051196671740952) q[0];
ry(0.835173036111664) q[1];
rz(2.085505199262258) q[1];
ry(-2.33766016849018) q[2];
rz(-2.486610821076387) q[2];
ry(-0.01102200126600005) q[3];
rz(-3.1078725235749705) q[3];
ry(-2.980334177857089) q[4];
rz(-2.2567845312644024) q[4];
ry(-0.10411389998102376) q[5];
rz(-0.08685699265192867) q[5];
ry(-3.1124215292633437) q[6];
rz(-1.1795715546509011) q[6];
ry(-0.09091351646824232) q[7];
rz(-0.4729374540695587) q[7];
ry(0.0034538998503740727) q[8];
rz(0.06870525582705689) q[8];
ry(-0.25067885826656555) q[9];
rz(-1.922936768547432) q[9];
ry(1.4158094720453553) q[10];
rz(-0.32576852643011467) q[10];
ry(1.1073074361373654) q[11];
rz(-2.6143007818774553) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.5880123338106761) q[0];
rz(2.094784176539283) q[0];
ry(-2.5471229015637866) q[1];
rz(2.67235530470579) q[1];
ry(-3.1280296918083086) q[2];
rz(-0.7243643699579408) q[2];
ry(-0.02508674359709673) q[3];
rz(0.8039559008532633) q[3];
ry(-1.1826511994073838) q[4];
rz(0.8263656915841464) q[4];
ry(3.1331336275308574) q[5];
rz(-2.5209495445693015) q[5];
ry(0.03344340346339881) q[6];
rz(2.1229419434729966) q[6];
ry(-1.5624355754971455) q[7];
rz(-2.344926167900287) q[7];
ry(-1.9407845028104038) q[8];
rz(-1.4095677780270384) q[8];
ry(2.2813406597751853) q[9];
rz(-2.81705349380572) q[9];
ry(1.6609485647150903) q[10];
rz(2.1214673812758043) q[10];
ry(-1.6107212017187358) q[11];
rz(-3.1171465009286043) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.07340105914784356) q[0];
rz(-2.4936820144336753) q[0];
ry(2.706771448374873) q[1];
rz(-0.44449796135918285) q[1];
ry(-0.7564762984835203) q[2];
rz(-1.5659803890452182) q[2];
ry(2.964427471705786) q[3];
rz(-2.2924731974910575) q[3];
ry(0.3074320525288485) q[4];
rz(2.6078642156021563) q[4];
ry(-1.9687837718337775) q[5];
rz(1.217341738233164) q[5];
ry(-0.026050486657528182) q[6];
rz(0.5640769793277066) q[6];
ry(-3.1186220222443906) q[7];
rz(0.6609437267377398) q[7];
ry(-3.1237354491677536) q[8];
rz(2.251034551234903) q[8];
ry(-0.003985945137125136) q[9];
rz(2.512654026607923) q[9];
ry(-2.327940504037968) q[10];
rz(2.03123319229745) q[10];
ry(-1.8832655788272759) q[11];
rz(-2.3073793173394885) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.224042963208361) q[0];
rz(1.1170266667381004) q[0];
ry(2.974847680124019) q[1];
rz(-0.07202450464152312) q[1];
ry(-3.1400417036496058) q[2];
rz(-1.8011510820158008) q[2];
ry(-0.017188584651188252) q[3];
rz(-2.761974261326487) q[3];
ry(3.031628256008127) q[4];
rz(2.2402404683951613) q[4];
ry(3.1275285997473357) q[5];
rz(2.7878777012492213) q[5];
ry(-0.005153144080546568) q[6];
rz(-2.333086027576863) q[6];
ry(0.33932029629742283) q[7];
rz(2.8809815476469303) q[7];
ry(0.5110214899289289) q[8];
rz(-2.435788606759901) q[8];
ry(2.2965436782905693) q[9];
rz(0.5658940372342329) q[9];
ry(-1.4773466375534168) q[10];
rz(0.07897766978416598) q[10];
ry(-1.838014073944323) q[11];
rz(-1.363093172682202) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.6100953369630275) q[0];
rz(0.04324263858776067) q[0];
ry(2.455534964906974) q[1];
rz(-2.369319204613617) q[1];
ry(2.460137435400601) q[2];
rz(-0.09694131669588124) q[2];
ry(3.035206164718253) q[3];
rz(2.1137084771620973) q[3];
ry(-2.509055867968243) q[4];
rz(-2.842383419525312) q[4];
ry(2.2342416388639603) q[5];
rz(-2.7372848617951666) q[5];
ry(0.021165430471534744) q[6];
rz(2.161694991687336) q[6];
ry(0.06008959830160343) q[7];
rz(-1.1938644346198777) q[7];
ry(-2.6946906070606853) q[8];
rz(2.857325779205817) q[8];
ry(-1.4820974341880249) q[9];
rz(0.031298952757845985) q[9];
ry(0.9765160128542494) q[10];
rz(2.273728158555567) q[10];
ry(0.042135637106104414) q[11];
rz(1.2680033116827438) q[11];