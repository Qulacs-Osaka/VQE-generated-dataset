OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.4612747208412413) q[0];
ry(1.2829146894932997) q[1];
cx q[0],q[1];
ry(-1.2646113664094942) q[0];
ry(0.8694968762982498) q[1];
cx q[0],q[1];
ry(-1.8557260139227205) q[2];
ry(-1.138443992920826) q[3];
cx q[2],q[3];
ry(1.8685309447063039) q[2];
ry(1.9210589037851915) q[3];
cx q[2],q[3];
ry(1.4819289126210453) q[1];
ry(-0.7218432887153521) q[2];
cx q[1],q[2];
ry(-1.559843454780509) q[1];
ry(2.99200790192649) q[2];
cx q[1],q[2];
ry(-2.8222360513133466) q[0];
ry(2.2484161210856812) q[1];
cx q[0],q[1];
ry(2.8787151181748767) q[0];
ry(-0.7234266301563806) q[1];
cx q[0],q[1];
ry(-2.920545248649692) q[2];
ry(0.5068500114008413) q[3];
cx q[2],q[3];
ry(-3.025400682633832) q[2];
ry(2.161984026861572) q[3];
cx q[2],q[3];
ry(1.6332322489808229) q[1];
ry(0.899927456469504) q[2];
cx q[1],q[2];
ry(-0.10204199688233562) q[1];
ry(-0.8065152444235327) q[2];
cx q[1],q[2];
ry(2.059021084653584) q[0];
ry(1.4040247637371168) q[1];
cx q[0],q[1];
ry(-3.1229248887097283) q[0];
ry(-1.2773303693185918) q[1];
cx q[0],q[1];
ry(-1.4247496001679334) q[2];
ry(0.7920793802249921) q[3];
cx q[2],q[3];
ry(0.17149806149914376) q[2];
ry(-0.6020119745921013) q[3];
cx q[2],q[3];
ry(-1.4630036290784973) q[1];
ry(1.220989626327241) q[2];
cx q[1],q[2];
ry(0.9486982736972155) q[1];
ry(0.36873743768749506) q[2];
cx q[1],q[2];
ry(-2.987500127820704) q[0];
ry(-1.2542532709226748) q[1];
ry(-1.2065994203340353) q[2];
ry(1.4012247905134556) q[3];