OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.25124449558428) q[0];
rz(0.5627149724607666) q[0];
ry(-0.8397876396095751) q[1];
rz(2.0314458294216937) q[1];
ry(0.38750350385445215) q[2];
rz(-0.9178476954069915) q[2];
ry(0.49943966194524236) q[3];
rz(-2.423763724984568) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.9906572677553859) q[0];
rz(-2.184629327819173) q[0];
ry(1.4189294958271992) q[1];
rz(1.0186885669395735) q[1];
ry(0.25424536516129415) q[2];
rz(3.1053758252840846) q[2];
ry(0.6344710476824911) q[3];
rz(2.796470889282821) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.25928878376579) q[0];
rz(0.11440628887726058) q[0];
ry(0.47748345055582764) q[1];
rz(-0.32298196534971435) q[1];
ry(0.6846022184501279) q[2];
rz(0.8470527444221204) q[2];
ry(-2.8651393446256694) q[3];
rz(-1.9580443949810828) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.4578516117887315) q[0];
rz(1.1663337405382057) q[0];
ry(-2.538210801105933) q[1];
rz(0.906539766329332) q[1];
ry(-2.5813744523891202) q[2];
rz(0.7100378280741632) q[2];
ry(-3.137090641776919) q[3];
rz(0.11005283638497686) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.818874858844297) q[0];
rz(-0.9835982311312035) q[0];
ry(0.7032017692962526) q[1];
rz(1.2211366012526588) q[1];
ry(1.3736966954679295) q[2];
rz(1.3984016557563814) q[2];
ry(-2.0663196969173194) q[3];
rz(-1.2181881389735154) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.20101319932660555) q[0];
rz(-0.328254179859322) q[0];
ry(3.1358187975156553) q[1];
rz(-2.1416024209553592) q[1];
ry(-2.832516753438582) q[2];
rz(-0.06598192620606333) q[2];
ry(2.22329670198973) q[3];
rz(3.1103681694416747) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.21215568089565) q[0];
rz(1.2691310174047692) q[0];
ry(3.013521065810343) q[1];
rz(0.40102379415715644) q[1];
ry(0.4641902595514349) q[2];
rz(-1.6010725604669265) q[2];
ry(-2.85509313293883) q[3];
rz(-0.46342364981130146) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.3960509641840181) q[0];
rz(1.3361760878023956) q[0];
ry(0.9076850478848498) q[1];
rz(1.5867166107694883) q[1];
ry(3.0471540958017242) q[2];
rz(1.9847332674993634) q[2];
ry(-2.585528057076073) q[3];
rz(0.2791720205483689) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.9102678846904975) q[0];
rz(1.6258704399829202) q[0];
ry(2.9747456851867327) q[1];
rz(-0.8955780282529647) q[1];
ry(1.8438281040406466) q[2];
rz(-1.388302370461342) q[2];
ry(-0.6211907517978766) q[3];
rz(-1.478015447470719) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.4424740521129076) q[0];
rz(3.0990729435558118) q[0];
ry(1.1630369230750377) q[1];
rz(-3.1167387466195966) q[1];
ry(-0.5054612524534923) q[2];
rz(-0.18265832011183022) q[2];
ry(-0.2254592255618328) q[3];
rz(-0.760960524105448) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.7288731472089307) q[0];
rz(-0.48665020981052437) q[0];
ry(2.1504888966005433) q[1];
rz(0.6282338825251063) q[1];
ry(0.6288967537251926) q[2];
rz(2.8855542258931126) q[2];
ry(0.9087989269718539) q[3];
rz(1.9469759358772059) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.8842512585950058) q[0];
rz(0.029281309926439295) q[0];
ry(-1.2506516546313036) q[1];
rz(-2.039657842866931) q[1];
ry(0.3868329532337773) q[2];
rz(1.1669932243750178) q[2];
ry(0.6513903767048986) q[3];
rz(-1.3605014766098247) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.20777221437947285) q[0];
rz(1.746483809338858) q[0];
ry(-2.8414469685926296) q[1];
rz(-0.22336513500883817) q[1];
ry(-2.20484836353037) q[2];
rz(0.5530778834405154) q[2];
ry(-2.5856986469306613) q[3];
rz(2.050843072336125) q[3];