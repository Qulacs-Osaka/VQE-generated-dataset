OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(3.1293386605885813) q[0];
ry(0.349646164100191) q[1];
cx q[0],q[1];
ry(1.4098758727754976) q[0];
ry(2.4806610051193303) q[1];
cx q[0],q[1];
ry(2.7910785596589855) q[1];
ry(1.579170121105002) q[2];
cx q[1],q[2];
ry(-2.4644241973927423) q[1];
ry(-0.06672279178072395) q[2];
cx q[1],q[2];
ry(0.9442398926732344) q[2];
ry(-2.614083832843535) q[3];
cx q[2],q[3];
ry(-0.6894076747984715) q[2];
ry(2.286682048122611) q[3];
cx q[2],q[3];
ry(0.6649944322696886) q[0];
ry(2.335134119726227) q[1];
cx q[0],q[1];
ry(0.9490940801181749) q[0];
ry(0.00010069324827345375) q[1];
cx q[0],q[1];
ry(2.2967276677893302) q[1];
ry(2.054888403137437) q[2];
cx q[1],q[2];
ry(-2.887354160601356) q[1];
ry(0.6740413326354687) q[2];
cx q[1],q[2];
ry(1.8202555044966315) q[2];
ry(2.67145471961702) q[3];
cx q[2],q[3];
ry(-1.691482673407539) q[2];
ry(-2.616298006296413) q[3];
cx q[2],q[3];
ry(0.5042371373242902) q[0];
ry(-0.1671881235672628) q[1];
cx q[0],q[1];
ry(-2.7266697427866706) q[0];
ry(-0.4497495164597087) q[1];
cx q[0],q[1];
ry(-2.4039759356096577) q[1];
ry(0.9325322874068783) q[2];
cx q[1],q[2];
ry(-0.4966696383110749) q[1];
ry(-0.06079995416175041) q[2];
cx q[1],q[2];
ry(-2.035465431685491) q[2];
ry(-2.271050445177314) q[3];
cx q[2],q[3];
ry(2.9462128711575213) q[2];
ry(-2.930765218687254) q[3];
cx q[2],q[3];
ry(-2.0869349248003504) q[0];
ry(-2.839401620174248) q[1];
ry(-1.8849606124599767) q[2];
ry(-2.3417862975070736) q[3];