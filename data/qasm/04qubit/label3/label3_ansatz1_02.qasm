OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(3.003268540539962e-06) q[0];
rz(-1.4365309026017428) q[0];
ry(2.1496843735647895) q[1];
rz(-1.728287799250234) q[1];
ry(-3.0037891118499713) q[2];
rz(1.9641012546531398) q[2];
ry(1.6748872809872015) q[3];
rz(2.4124135989889126) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.617502478064821) q[0];
rz(-3.118129099054511) q[0];
ry(1.846048806001285) q[1];
rz(1.3262966225440183) q[1];
ry(1.5687982598737409) q[2];
rz(2.2666699887772745) q[2];
ry(-0.25647153673257606) q[3];
rz(0.057504955629074324) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.5835579899606875) q[0];
rz(-1.8672751522406514) q[0];
ry(-4.171446299494619e-08) q[1];
rz(-1.069457757165507) q[1];
ry(1.6573399415480228) q[2];
rz(-1.2740887072805265) q[2];
ry(0.06825660622190899) q[3];
rz(-0.622288284046592) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.3716043168679204) q[0];
rz(1.001543098114774) q[0];
ry(-1.5633525960573729) q[1];
rz(1.4690222227538665) q[1];
ry(3.130452839835029) q[2];
rz(-0.5226653494827778) q[2];
ry(-0.1017725625305602) q[3];
rz(-2.8938760020111958) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.9762514564362812) q[0];
rz(1.4690640761509506) q[0];
ry(-1.6334890531234978) q[1];
rz(2.555241085148474) q[1];
ry(-0.10189053899334422) q[2];
rz(0.43319523172005603) q[2];
ry(-0.9852238672165248) q[3];
rz(-0.13403350574881165) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.3891313137760946) q[0];
rz(1.6423968757135778) q[0];
ry(-0.3609600978800822) q[1];
rz(-2.515777794166252) q[1];
ry(-2.9435911948183193) q[2];
rz(1.1502239599194304) q[2];
ry(1.1712863136078138) q[3];
rz(3.1324741827134206) q[3];