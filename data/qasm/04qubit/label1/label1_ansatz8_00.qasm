OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.2903382008249618) q[0];
ry(-0.2670783821787186) q[1];
cx q[0],q[1];
ry(1.214378149939889) q[0];
ry(1.4966454240727018) q[1];
cx q[0],q[1];
ry(1.4739697678683579) q[2];
ry(-0.3342571813044537) q[3];
cx q[2],q[3];
ry(2.1129881332988063) q[2];
ry(2.87960886414327) q[3];
cx q[2],q[3];
ry(0.5418373270056344) q[0];
ry(-1.2805433983348316) q[2];
cx q[0],q[2];
ry(0.5824300268617231) q[0];
ry(-0.3136922982875774) q[2];
cx q[0],q[2];
ry(-1.6242514901300584) q[1];
ry(3.005219716509366) q[3];
cx q[1],q[3];
ry(-1.9640826639285862) q[1];
ry(-3.0969509165546723) q[3];
cx q[1],q[3];
ry(2.0244976228238087) q[0];
ry(-3.073645814851179) q[1];
cx q[0],q[1];
ry(-2.4023838222763394) q[0];
ry(-2.1323826587952883) q[1];
cx q[0],q[1];
ry(-1.3496457877978951) q[2];
ry(2.932377248318309) q[3];
cx q[2],q[3];
ry(-1.016908941111855) q[2];
ry(0.4988438632186831) q[3];
cx q[2],q[3];
ry(1.4734241052620245) q[0];
ry(-0.05775015886902146) q[2];
cx q[0],q[2];
ry(-0.3976147206094378) q[0];
ry(-2.665235496825517) q[2];
cx q[0],q[2];
ry(2.202569758749942) q[1];
ry(2.603149330093783) q[3];
cx q[1],q[3];
ry(0.5234513126909421) q[1];
ry(-0.7026773870049885) q[3];
cx q[1],q[3];
ry(0.3146610203384739) q[0];
ry(0.634049163192425) q[1];
cx q[0],q[1];
ry(-0.9577217102720442) q[0];
ry(0.6282851011429731) q[1];
cx q[0],q[1];
ry(-0.8571336166856166) q[2];
ry(-0.8684960820638779) q[3];
cx q[2],q[3];
ry(-0.6296783813202909) q[2];
ry(-0.42440535476586316) q[3];
cx q[2],q[3];
ry(-0.04729756253955574) q[0];
ry(-1.552337203068169) q[2];
cx q[0],q[2];
ry(-0.7915994220706339) q[0];
ry(2.303740647655383) q[2];
cx q[0],q[2];
ry(0.0860679270626317) q[1];
ry(2.1066601058546928) q[3];
cx q[1],q[3];
ry(-1.592645034449367) q[1];
ry(-2.418538781979899) q[3];
cx q[1],q[3];
ry(-2.5926300885721885) q[0];
ry(2.7628841158662736) q[1];
ry(-3.093701390021622) q[2];
ry(0.3285038080240233) q[3];