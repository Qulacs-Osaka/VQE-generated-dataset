OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.5854234779643583) q[0];
ry(2.9171150881853043) q[1];
cx q[0],q[1];
ry(2.3545571302140464) q[0];
ry(-1.2398754430480017) q[1];
cx q[0],q[1];
ry(2.4127463895315686) q[2];
ry(0.0249108960428387) q[3];
cx q[2],q[3];
ry(0.9865599497526008) q[2];
ry(-2.9194830312142055) q[3];
cx q[2],q[3];
ry(-2.6316329140353143) q[0];
ry(0.034162372852137864) q[2];
cx q[0],q[2];
ry(2.453627086929108) q[0];
ry(-0.3133622242788139) q[2];
cx q[0],q[2];
ry(-1.686245792253441) q[1];
ry(2.529765427142258) q[3];
cx q[1],q[3];
ry(-0.6204510768369467) q[1];
ry(1.5794309028725722) q[3];
cx q[1],q[3];
ry(1.412734841689783) q[0];
ry(2.5357798177406927) q[1];
cx q[0],q[1];
ry(-2.441435871311691) q[0];
ry(2.8744203220325204) q[1];
cx q[0],q[1];
ry(-1.7231777190029094) q[2];
ry(1.713824757948407) q[3];
cx q[2],q[3];
ry(-1.6785035856579196) q[2];
ry(-0.744724601457101) q[3];
cx q[2],q[3];
ry(0.3451349931110284) q[0];
ry(-1.7300892895519857) q[2];
cx q[0],q[2];
ry(1.3107870931319634) q[0];
ry(-0.003889978211575773) q[2];
cx q[0],q[2];
ry(0.30460918985597557) q[1];
ry(0.6281945800547649) q[3];
cx q[1],q[3];
ry(1.9396129031078884) q[1];
ry(1.401552199545125) q[3];
cx q[1],q[3];
ry(1.8167109061072226) q[0];
ry(0.5055901240399416) q[1];
cx q[0],q[1];
ry(3.0604507935260488) q[0];
ry(1.9572817918553502) q[1];
cx q[0],q[1];
ry(-0.10954065324275436) q[2];
ry(-1.877673820646784) q[3];
cx q[2],q[3];
ry(-1.5128148257223282) q[2];
ry(-2.506767267626754) q[3];
cx q[2],q[3];
ry(-1.7803769607658442) q[0];
ry(2.783038536693399) q[2];
cx q[0],q[2];
ry(-1.855170824980538) q[0];
ry(-1.1636304391366608) q[2];
cx q[0],q[2];
ry(0.9824528921139244) q[1];
ry(0.39129158335077174) q[3];
cx q[1],q[3];
ry(-1.1234456906276034) q[1];
ry(1.0746254386006935) q[3];
cx q[1],q[3];
ry(1.0141100768759275) q[0];
ry(1.653724047025034) q[1];
cx q[0],q[1];
ry(-1.3141637800807917) q[0];
ry(-1.6679064184889132) q[1];
cx q[0],q[1];
ry(-2.50092282633595) q[2];
ry(-2.111867896977363) q[3];
cx q[2],q[3];
ry(-1.206087200131475) q[2];
ry(-0.6780399828261796) q[3];
cx q[2],q[3];
ry(2.6325925392499943) q[0];
ry(1.899143097153031) q[2];
cx q[0],q[2];
ry(1.9493079056330547) q[0];
ry(3.0196236223420305) q[2];
cx q[0],q[2];
ry(0.9917856680248862) q[1];
ry(-1.3674882243573467) q[3];
cx q[1],q[3];
ry(-2.4627385347900685) q[1];
ry(-2.1614813351606275) q[3];
cx q[1],q[3];
ry(2.5434672827808305) q[0];
ry(-1.6676596116634266) q[1];
cx q[0],q[1];
ry(0.9858398588472026) q[0];
ry(0.80057916145172) q[1];
cx q[0],q[1];
ry(-1.2290886285258948) q[2];
ry(-2.687698532645157) q[3];
cx q[2],q[3];
ry(1.6232185249953994) q[2];
ry(0.9833285337875273) q[3];
cx q[2],q[3];
ry(-1.9314594790249684) q[0];
ry(-2.4928415351305255) q[2];
cx q[0],q[2];
ry(-0.1038204725660428) q[0];
ry(1.0104298206002251) q[2];
cx q[0],q[2];
ry(-2.3666551729867202) q[1];
ry(2.419502500488116) q[3];
cx q[1],q[3];
ry(-2.2742735207224074) q[1];
ry(1.666994067315768) q[3];
cx q[1],q[3];
ry(0.869733340164741) q[0];
ry(-2.7903986387327304) q[1];
cx q[0],q[1];
ry(1.8671089152738132) q[0];
ry(2.354130238427265) q[1];
cx q[0],q[1];
ry(2.424552441356784) q[2];
ry(-2.249892090552991) q[3];
cx q[2],q[3];
ry(-3.0947648762232207) q[2];
ry(-2.2094153524980102) q[3];
cx q[2],q[3];
ry(-1.3167922804342567) q[0];
ry(1.7553584630412224) q[2];
cx q[0],q[2];
ry(-0.4766523463663832) q[0];
ry(2.962191773265759) q[2];
cx q[0],q[2];
ry(2.6479971963292264) q[1];
ry(0.4916874637564854) q[3];
cx q[1],q[3];
ry(-2.2164511928801707) q[1];
ry(2.0659623727283045) q[3];
cx q[1],q[3];
ry(-1.046203828549298) q[0];
ry(1.7275179349911907) q[1];
cx q[0],q[1];
ry(-0.6573137856225233) q[0];
ry(-0.8624310883881421) q[1];
cx q[0],q[1];
ry(2.5565500265567414) q[2];
ry(-2.4599829312504307) q[3];
cx q[2],q[3];
ry(-2.3088015533725703) q[2];
ry(-2.668722084103641) q[3];
cx q[2],q[3];
ry(-1.248848334566514) q[0];
ry(-2.201044272284036) q[2];
cx q[0],q[2];
ry(1.451831340898603) q[0];
ry(0.41667734105483856) q[2];
cx q[0],q[2];
ry(3.0089583352708096) q[1];
ry(-0.4579270971607788) q[3];
cx q[1],q[3];
ry(2.5827553317361005) q[1];
ry(-2.7253822229564526) q[3];
cx q[1],q[3];
ry(-2.1997585920916065) q[0];
ry(0.28059498753409545) q[1];
ry(1.733175911002678) q[2];
ry(-1.1006631032521472) q[3];