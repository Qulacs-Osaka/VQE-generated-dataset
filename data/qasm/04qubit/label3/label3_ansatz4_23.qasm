OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(2.056120255051068) q[0];
rz(-1.4085640407855369) q[0];
ry(2.418123338401877) q[1];
rz(-0.5975610491088297) q[1];
ry(1.3955121921572182) q[2];
rz(-0.3287141410107685) q[2];
ry(1.308591042838202) q[3];
rz(-0.7247902156734197) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.012710168632514477) q[0];
rz(-2.6702161053308644) q[0];
ry(-2.471630180588897) q[1];
rz(-0.8380423510153148) q[1];
ry(-0.8597108595072945) q[2];
rz(0.027306675586504903) q[2];
ry(-2.5598676463695553) q[3];
rz(1.1738158092505806) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.5405360345135177) q[0];
rz(-1.218662480558648) q[0];
ry(1.5446604549311767) q[1];
rz(-1.9229041640194735) q[1];
ry(-2.0265874063924967) q[2];
rz(0.50630673382055) q[2];
ry(2.101650987850074) q[3];
rz(0.0298203905198154) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.1415556835177298) q[0];
rz(0.6630025181639324) q[0];
ry(-1.3214808417871584) q[1];
rz(1.5816200465835162) q[1];
ry(-2.744162860880714) q[2];
rz(-2.3263128424705313) q[2];
ry(-2.2989122709575454) q[3];
rz(1.5131840262551102) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.6233051432116437) q[0];
rz(2.585340484915873) q[0];
ry(2.178854848231489) q[1];
rz(1.0904112024708486) q[1];
ry(0.4488022975515707) q[2];
rz(-0.29912561437123747) q[2];
ry(0.2249636427063777) q[3];
rz(-2.3477316514087687) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(3.0142851814275566) q[0];
rz(2.205615514940012) q[0];
ry(0.18352463539766184) q[1];
rz(1.7329036538549698) q[1];
ry(1.590135125671462) q[2];
rz(-2.843610942310097) q[2];
ry(-1.9443530500836275) q[3];
rz(3.1172263442994765) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(3.0327748211698378) q[0];
rz(-0.24326036302490195) q[0];
ry(-1.1417517682122558) q[1];
rz(0.06902707798496252) q[1];
ry(0.38218131741720013) q[2];
rz(0.8554105607968359) q[2];
ry(-0.9694000520869839) q[3];
rz(3.038342536018996) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.8488755499589775) q[0];
rz(-1.4988096395521953) q[0];
ry(-2.8776342467647202) q[1];
rz(0.9825669553442233) q[1];
ry(2.1726664668953353) q[2];
rz(-2.978830338731612) q[2];
ry(1.6645983513278368) q[3];
rz(0.4943477299518335) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.7154066559920355) q[0];
rz(1.314011063697425) q[0];
ry(1.3444374536160915) q[1];
rz(2.265962909291247) q[1];
ry(-0.42078436846969514) q[2];
rz(2.79991538391349) q[2];
ry(0.35819485340904855) q[3];
rz(-2.80006254955054) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.4298361469814314) q[0];
rz(0.10651923262961026) q[0];
ry(0.10948145160372103) q[1];
rz(1.7465200774165242) q[1];
ry(-1.8710209266209077) q[2];
rz(1.809545123320374) q[2];
ry(2.8140333390811954) q[3];
rz(-2.51674198028584) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.6975964430719965) q[0];
rz(-0.05896025749287404) q[0];
ry(1.1774524782306104) q[1];
rz(-1.9558494073731287) q[1];
ry(2.1973288380665528) q[2];
rz(2.9882884575072293) q[2];
ry(-2.5609485359567143) q[3];
rz(-2.229834481088181) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.8698384776919426) q[0];
rz(0.0575048332616064) q[0];
ry(1.8897279034855465) q[1];
rz(3.0857411123354948) q[1];
ry(1.5565810044160777) q[2];
rz(1.0039380254757784) q[2];
ry(2.506494123686671) q[3];
rz(1.015597272003276) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.6331817261724315) q[0];
rz(2.697177787260603) q[0];
ry(-1.9037566512759057) q[1];
rz(-1.8167184096759221) q[1];
ry(-2.450954076844832) q[2];
rz(-2.22991347801909) q[2];
ry(-1.2949318381987305) q[3];
rz(2.663720308632273) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.038893197600046) q[0];
rz(0.7909200037457265) q[0];
ry(-0.387922402038313) q[1];
rz(1.2197612658698336) q[1];
ry(0.0323729089876874) q[2];
rz(0.1906660932496668) q[2];
ry(-2.8455421030017773) q[3];
rz(-1.8301989719274818) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.8808385530964093) q[0];
rz(-1.8021647355760075) q[0];
ry(2.7128428638338082) q[1];
rz(-2.076519330042065) q[1];
ry(0.39765092650396383) q[2];
rz(1.2421668728613098) q[2];
ry(2.6216277035147875) q[3];
rz(0.46137937773786936) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.3151554353641446) q[0];
rz(-2.2880378151099126) q[0];
ry(2.982713771768266) q[1];
rz(2.761830281403873) q[1];
ry(1.9536266691555115) q[2];
rz(-2.7126641437553274) q[2];
ry(-2.174635602469233) q[3];
rz(-1.5075953094828733) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.456127762496224) q[0];
rz(-0.7994035682810177) q[0];
ry(-2.7616756698319573) q[1];
rz(0.2331512070906818) q[1];
ry(1.3181941282708292) q[2];
rz(-2.3131749022498687) q[2];
ry(1.4015789128460836) q[3];
rz(0.2518901606030069) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.4996166788647265) q[0];
rz(-1.9688984010388344) q[0];
ry(0.3716280395364945) q[1];
rz(2.6417935446233023) q[1];
ry(-0.5106709737354314) q[2];
rz(2.1224661664283757) q[2];
ry(1.1862648713159973) q[3];
rz(-2.303009647105147) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.6712760474047903) q[0];
rz(0.09336046241865868) q[0];
ry(-0.06596527183258517) q[1];
rz(1.1043232554557) q[1];
ry(2.5270043250238126) q[2];
rz(1.791333063154839) q[2];
ry(-2.534105979482557) q[3];
rz(-0.010454653504241294) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.4218832686733394) q[0];
rz(2.8288568869960384) q[0];
ry(-2.5851711699429445) q[1];
rz(-0.013218745460471306) q[1];
ry(-1.7655596784680074) q[2];
rz(0.3880143794794433) q[2];
ry(0.9017141214735362) q[3];
rz(-1.8643567712760096) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.8359155154680336) q[0];
rz(-0.09349502102446115) q[0];
ry(3.1208329851352588) q[1];
rz(-1.338871768334001) q[1];
ry(-2.7563193723687895) q[2];
rz(-1.1814337623352502) q[2];
ry(-1.5081063619227022) q[3];
rz(0.6222730998655441) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.356093837091266) q[0];
rz(-1.5551968508375822) q[0];
ry(2.6633211471457368) q[1];
rz(-2.3552386368197467) q[1];
ry(1.0206873032007673) q[2];
rz(1.0765214884583694) q[2];
ry(-0.13446134897853135) q[3];
rz(1.6293241302554933) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.208920527798453) q[0];
rz(-1.452889604130213) q[0];
ry(-2.253670409843685) q[1];
rz(0.41423449068526025) q[1];
ry(0.8738116427272091) q[2];
rz(2.4042825793124774) q[2];
ry(-1.803295433672113) q[3];
rz(-0.38293940581184543) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.7773783109061767) q[0];
rz(-1.2521811661747542) q[0];
ry(2.0493702159281737) q[1];
rz(0.987554456256235) q[1];
ry(-0.8385457960864199) q[2];
rz(2.749242686799546) q[2];
ry(-2.157136448726721) q[3];
rz(-0.17349226756597336) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.3305006032412239) q[0];
rz(0.07242793518426818) q[0];
ry(0.5843216229920465) q[1];
rz(-1.712794149393571) q[1];
ry(0.0346498918810747) q[2];
rz(0.07927632865600387) q[2];
ry(-0.5985193003537099) q[3];
rz(-0.25485083001677555) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.0395400406145505) q[0];
rz(-2.7787150960289324) q[0];
ry(3.0017552106758396) q[1];
rz(-1.8393777564220954) q[1];
ry(-1.5065899586448293) q[2];
rz(0.7917266847422074) q[2];
ry(-1.9196463868280644) q[3];
rz(2.6307481721918267) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.3103757814520212) q[0];
rz(-2.1623151733660393) q[0];
ry(3.1328698037059977) q[1];
rz(-1.3232116221297585) q[1];
ry(0.025756029914206557) q[2];
rz(-2.6601950955402827) q[2];
ry(2.722357330584667) q[3];
rz(0.7884498878327275) q[3];