OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.8601308274005648) q[0];
rz(0.23819707484023994) q[0];
ry(-0.9332083402279743) q[1];
rz(0.7859875990187932) q[1];
ry(-0.9696283379992265) q[2];
rz(-0.9007736611493025) q[2];
ry(0.36527679849827066) q[3];
rz(-0.6856511266164415) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.9925426114493443) q[0];
rz(-0.6784283144882801) q[0];
ry(-2.8331862204577356) q[1];
rz(-0.45364764497355664) q[1];
ry(-0.2927496185625889) q[2];
rz(0.3598456981751355) q[2];
ry(-1.8237809202017474) q[3];
rz(-0.2995194539163153) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.2787755432424524) q[0];
rz(2.761402957276156) q[0];
ry(-0.9827325054916249) q[1];
rz(-2.0992069584892317) q[1];
ry(0.7167308699701309) q[2];
rz(-0.17269280253069616) q[2];
ry(-1.0053800643705602) q[3];
rz(-0.11218628166247767) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-3.104583827518058) q[0];
rz(-0.8026711763781966) q[0];
ry(-1.7015922178245764) q[1];
rz(2.85765794994281) q[1];
ry(-0.5588446809263278) q[2];
rz(2.365474247952961) q[2];
ry(2.555621513198001) q[3];
rz(-1.8282516611653206) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.44341882986537007) q[0];
rz(2.7010853213418775) q[0];
ry(0.24095363983393447) q[1];
rz(-2.2635175422292813) q[1];
ry(2.982863261646951) q[2];
rz(0.06308462350358306) q[2];
ry(-1.1931896829057949) q[3];
rz(-2.239913837853324) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.41050207036267466) q[0];
rz(-2.848097907701703) q[0];
ry(0.32492164631238674) q[1];
rz(-0.9246119163661406) q[1];
ry(-2.9265962678657256) q[2];
rz(-2.039104589927429) q[2];
ry(-1.7011339982205227) q[3];
rz(0.4183452337476304) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.39050428520040814) q[0];
rz(-0.005027869070307389) q[0];
ry(-2.0424466186424657) q[1];
rz(-1.3681012370118957) q[1];
ry(2.684741875076739) q[2];
rz(0.18831863489398482) q[2];
ry(-2.5580559357855477) q[3];
rz(2.365975389326317) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.8878863118749336) q[0];
rz(-1.295383229767188) q[0];
ry(2.54556721176525) q[1];
rz(-0.9425007982193145) q[1];
ry(-0.1496219788373576) q[2];
rz(-1.3249465655500634) q[2];
ry(-0.5323551875922279) q[3];
rz(2.0004820543599804) q[3];