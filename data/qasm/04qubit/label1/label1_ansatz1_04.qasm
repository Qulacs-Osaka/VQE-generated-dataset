OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.24147554992595377) q[0];
rz(-0.2806360003295866) q[0];
ry(2.668028356621488) q[1];
rz(-0.5679769494327225) q[1];
ry(2.955179133537611) q[2];
rz(-2.8594480763922565) q[2];
ry(1.6509614392950784) q[3];
rz(2.7151608074548137) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.48726726244462126) q[0];
rz(1.744466086040974) q[0];
ry(1.0642440197464742) q[1];
rz(2.1798708095351067) q[1];
ry(-1.8060990296559203) q[2];
rz(-0.47893238613049194) q[2];
ry(0.8680857262176556) q[3];
rz(-1.3692156731618415) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.3136939552584945) q[0];
rz(-2.192607623432866) q[0];
ry(2.6232710513387336) q[1];
rz(-2.469289285953109) q[1];
ry(2.125078693109457) q[2];
rz(1.6415756047804058) q[2];
ry(2.945557717187339) q[3];
rz(-2.0220900889556717) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.8589757894200778) q[0];
rz(-0.6915172541899115) q[0];
ry(-1.4202960099772737) q[1];
rz(0.013274482196906412) q[1];
ry(-1.2754985038424067) q[2];
rz(1.4666714784473762) q[2];
ry(-2.1777385283444897) q[3];
rz(1.6364650491373212) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.0101602918417) q[0];
rz(-2.013121152691793) q[0];
ry(1.2803988029337463) q[1];
rz(-2.6535223806533437) q[1];
ry(-0.5318830477251741) q[2];
rz(2.644401390585058) q[2];
ry(3.0318899158277115) q[3];
rz(-0.5006945807193989) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.452521623726976) q[0];
rz(2.081030554203213) q[0];
ry(-1.3202143484415423) q[1];
rz(-1.556016337316799) q[1];
ry(1.4719738947442504) q[2];
rz(-2.3971742235660236) q[2];
ry(-2.8934162252247697) q[3];
rz(-0.49729278143719735) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.587217833862013) q[0];
rz(2.910379700809049) q[0];
ry(-0.07420986158028509) q[1];
rz(-2.690162717275459) q[1];
ry(-0.054441960518818895) q[2];
rz(0.7419417994649775) q[2];
ry(-1.0084592547070619) q[3];
rz(-2.0045549031585908) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.499212167439669) q[0];
rz(1.7224828205601406) q[0];
ry(-1.1020176926179528) q[1];
rz(0.38435261075507465) q[1];
ry(-2.0386464316656294) q[2];
rz(-1.5664758274852648) q[2];
ry(-1.5785470399125607) q[3];
rz(-0.618313126523522) q[3];