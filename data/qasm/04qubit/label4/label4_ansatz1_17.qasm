OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.7575527754487643) q[0];
rz(1.5024467384382467) q[0];
ry(2.2474289193510937) q[1];
rz(0.9440793741923339) q[1];
ry(1.0957493340430557) q[2];
rz(-1.9843141806830396) q[2];
ry(2.120516724043657) q[3];
rz(-2.073632123977815) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.182670354111236) q[0];
rz(1.5164713388207203) q[0];
ry(-2.3405551959367665) q[1];
rz(1.3939381592318432) q[1];
ry(-0.09489066337142127) q[2];
rz(0.6889692930008476) q[2];
ry(-1.358563144677901) q[3];
rz(0.5312566417078172) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.12465927662380058) q[0];
rz(-0.7299234498219596) q[0];
ry(-0.6512444437228605) q[1];
rz(2.949964017446163) q[1];
ry(2.0087469639584685) q[2];
rz(-0.8189212467166184) q[2];
ry(-1.1793903573559144) q[3];
rz(-0.9462349047295224) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.4524381004032821) q[0];
rz(-1.1778637196531188) q[0];
ry(0.7603551134032323) q[1];
rz(-1.5328685101681094) q[1];
ry(-2.5301158171611347) q[2];
rz(2.69605307904879) q[2];
ry(0.16432221721579945) q[3];
rz(-2.6168003164476588) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.910866412614764) q[0];
rz(0.3340790520992858) q[0];
ry(-1.6646462143209848) q[1];
rz(0.3039911732993348) q[1];
ry(0.8438918392459627) q[2];
rz(0.9529007748570044) q[2];
ry(0.08989872872357729) q[3];
rz(-2.9346295118417793) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.5794119231445216) q[0];
rz(3.0675100783444336) q[0];
ry(-0.045837262980171545) q[1];
rz(0.6421201213510931) q[1];
ry(2.7125479352159516) q[2];
rz(-1.1673456475576103) q[2];
ry(2.882612054354748) q[3];
rz(0.44477489795720265) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.3268588227516043) q[0];
rz(0.35586284039030947) q[0];
ry(0.5116497320706271) q[1];
rz(-0.23553078650676174) q[1];
ry(0.6560422166944947) q[2];
rz(0.617477497649639) q[2];
ry(-1.1330174062475882) q[3];
rz(0.03793230300669315) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.7932223551306476) q[0];
rz(-1.439033460076864) q[0];
ry(-2.66108431247054) q[1];
rz(1.900764113763934) q[1];
ry(-0.9475794585986765) q[2];
rz(2.332471661110531) q[2];
ry(0.0009093154898509898) q[3];
rz(0.941412946123729) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.5128150305387689) q[0];
rz(0.8278918613422794) q[0];
ry(1.4187399159379694) q[1];
rz(1.3603943683061122) q[1];
ry(-0.20625019674706557) q[2];
rz(-1.8257102402310283) q[2];
ry(2.421648991485817) q[3];
rz(-2.7773446773789745) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.8357549617707036) q[0];
rz(2.797481271217774) q[0];
ry(1.3679951996771398) q[1];
rz(2.8423880304230877) q[1];
ry(-2.5513565901994393) q[2];
rz(-0.8248642028736989) q[2];
ry(-2.208428642304142) q[3];
rz(0.179676848557306) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.4322747206562391) q[0];
rz(1.5405991875735483) q[0];
ry(1.4245813245265477) q[1];
rz(-1.9857711751019709) q[1];
ry(0.2551374165986031) q[2];
rz(-1.8872449687657329) q[2];
ry(2.915122724109316) q[3];
rz(2.796573137879186) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.865666370863319) q[0];
rz(-3.086737514808569) q[0];
ry(-1.8874091194044298) q[1];
rz(2.2970261702564976) q[1];
ry(1.4464301496071474) q[2];
rz(-2.3644978251297184) q[2];
ry(-2.625135718509203) q[3];
rz(-2.491549619205486) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.8177753606169597) q[0];
rz(-2.9837235708783276) q[0];
ry(3.0289179520356115) q[1];
rz(1.671024420037198) q[1];
ry(1.3128881160859374) q[2];
rz(1.9366507626136962) q[2];
ry(-0.03212210462601946) q[3];
rz(-1.6209687936581045) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.15147197955076933) q[0];
rz(-0.4678383500459438) q[0];
ry(-0.4770413906129968) q[1];
rz(-1.6227378898482503) q[1];
ry(0.33190253225246824) q[2];
rz(-3.1014214897595163) q[2];
ry(-2.361229685127854) q[3];
rz(1.3428121997331395) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.9911301649042354) q[0];
rz(-0.4201527588352869) q[0];
ry(-2.293723665345671) q[1];
rz(2.4910933842676624) q[1];
ry(-1.056102818748992) q[2];
rz(-0.20380161985861098) q[2];
ry(-3.1396257254157636) q[3];
rz(-0.6151670369541415) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.14028688875700787) q[0];
rz(-3.0060803770623883) q[0];
ry(0.6184304308930733) q[1];
rz(-0.11583392388069047) q[1];
ry(-0.2995389970225011) q[2];
rz(-2.133142143064891) q[2];
ry(-0.2927373487702799) q[3];
rz(-2.8582974296382826) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.08747845246280671) q[0];
rz(2.808133295970843) q[0];
ry(-2.665494409385095) q[1];
rz(-2.683761352465735) q[1];
ry(-0.7824005120101513) q[2];
rz(-1.8814006393905762) q[2];
ry(2.6099202298488544) q[3];
rz(-2.362014764126288) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.6321908108411094) q[0];
rz(-0.8299823878210392) q[0];
ry(2.0328629602053456) q[1];
rz(1.287983959467703) q[1];
ry(3.077168181417078) q[2];
rz(-1.2648834877213717) q[2];
ry(1.1398405810456271) q[3];
rz(-2.0091403879994765) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.2739024451458718) q[0];
rz(1.0507306510001704) q[0];
ry(-2.7511829749414054) q[1];
rz(-1.6558774625960313) q[1];
ry(-1.7089993150586347) q[2];
rz(-0.7145395601022287) q[2];
ry(2.033196321360886) q[3];
rz(3.0163391911994992) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.7768985843543383) q[0];
rz(-2.3657927407486072) q[0];
ry(1.4105316771384135) q[1];
rz(1.7374331417122562) q[1];
ry(-1.0873275849980935) q[2];
rz(2.4812667946046982) q[2];
ry(0.7492527067569478) q[3];
rz(-1.8025256481895107) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.0918154226459447) q[0];
rz(1.1254031734311443) q[0];
ry(1.7601183711166586) q[1];
rz(0.11721717423629686) q[1];
ry(2.4473030053238167) q[2];
rz(-0.5864579082013167) q[2];
ry(0.5169236418933444) q[3];
rz(-1.8048821599730953) q[3];