OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.8783207116677794) q[0];
rz(2.4205389095397667) q[0];
ry(-0.13135162963757274) q[1];
rz(1.2575850490789715) q[1];
ry(-1.9598781618259968) q[2];
rz(-2.219940165814636) q[2];
ry(1.069364473142698) q[3];
rz(0.8916392888705928) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.09945065874347588) q[0];
rz(0.3701416405193943) q[0];
ry(0.03202787902455724) q[1];
rz(1.7755831980508736) q[1];
ry(2.1504659680243754) q[2];
rz(-2.2405772296082738) q[2];
ry(0.8870708758151349) q[3];
rz(1.7411772429743735) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.288898393526258) q[0];
rz(2.405590787194648) q[0];
ry(-2.6672215888358606) q[1];
rz(-1.2961139373521746) q[1];
ry(-0.40478650595169097) q[2];
rz(0.165758771284354) q[2];
ry(1.718406629698202) q[3];
rz(-1.2312257899010752) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.4158189024804516) q[0];
rz(2.7710541170413383) q[0];
ry(0.6101641398784307) q[1];
rz(0.06879363015960925) q[1];
ry(1.525789253838771) q[2];
rz(2.5431974196299523) q[2];
ry(1.044556464170363) q[3];
rz(-2.391866168033847) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.392233057622792) q[0];
rz(-2.7204239289456225) q[0];
ry(-1.249549686829648) q[1];
rz(2.801218267323042) q[1];
ry(2.2958726789977955) q[2];
rz(0.6154765447878586) q[2];
ry(2.0172444839012837) q[3];
rz(0.5578201851742889) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.9696732227963942) q[0];
rz(0.5550015745597977) q[0];
ry(0.06578777351011082) q[1];
rz(0.8415555329975709) q[1];
ry(3.1211308000148055) q[2];
rz(-1.9868180347903577) q[2];
ry(-1.1859012957161856) q[3];
rz(-0.2333195016076033) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.9430841445038265) q[0];
rz(2.8657820779546235) q[0];
ry(-0.6568264508928863) q[1];
rz(0.5633719039760404) q[1];
ry(-1.464962471354566) q[2];
rz(2.5634767560544782) q[2];
ry(-2.201145272235431) q[3];
rz(-0.8332918106431442) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.1601745734394084) q[0];
rz(-0.8919960902882957) q[0];
ry(0.30990774342457783) q[1];
rz(1.338662366112504) q[1];
ry(0.7551779728481893) q[2];
rz(-3.031411363118125) q[2];
ry(2.061713825659451) q[3];
rz(-0.535318448132438) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.3073828428075434) q[0];
rz(1.38091080850776) q[0];
ry(-1.8178069468065585) q[1];
rz(-2.410003702468654) q[1];
ry(-1.8294256692694792) q[2];
rz(2.483549870527565) q[2];
ry(-2.2711285617562362) q[3];
rz(-0.0801600308549589) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.3306892263760117) q[0];
rz(2.379589449795581) q[0];
ry(2.918522433918255) q[1];
rz(-0.9854211039224018) q[1];
ry(-1.1209916788375116) q[2];
rz(-1.2419699523289247) q[2];
ry(2.4305605225804983) q[3];
rz(2.075880685943986) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.6482142901009267) q[0];
rz(-2.460147791816759) q[0];
ry(-2.829705936297377) q[1];
rz(2.5720818482805354) q[1];
ry(-1.2120938247516773) q[2];
rz(1.882893274821753) q[2];
ry(0.5541945490906983) q[3];
rz(-2.0829072009360434) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.6173221301964795) q[0];
rz(0.2677448210296198) q[0];
ry(-0.8380422925267395) q[1];
rz(3.122452467406972) q[1];
ry(-0.35222357388807257) q[2];
rz(-2.3887703278680448) q[2];
ry(-2.9951013913294973) q[3];
rz(-0.773421715844507) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.6978748059443985) q[0];
rz(2.1527759954566807) q[0];
ry(-1.6990710417569068) q[1];
rz(0.7728445930515223) q[1];
ry(0.7342631801971025) q[2];
rz(1.8223463570895029) q[2];
ry(0.10567845025050054) q[3];
rz(-2.9618269854965775) q[3];