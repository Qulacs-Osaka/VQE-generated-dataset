OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.295488141721201) q[0];
rz(-2.898678704082924) q[0];
ry(-0.0885593785308849) q[1];
rz(0.6060572076275443) q[1];
ry(0.9329362046023536) q[2];
rz(-2.6025705723268695) q[2];
ry(-0.8950841865232171) q[3];
rz(1.6084179172205637) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.0305133020024144) q[0];
rz(-0.5768245347967832) q[0];
ry(2.9535830452202014) q[1];
rz(-0.20133930744017814) q[1];
ry(-1.7496915515421607) q[2];
rz(-1.4146069922172744) q[2];
ry(-0.11928057536056007) q[3];
rz(1.6134312594483298) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.8716300952919833) q[0];
rz(-2.4432937700471946) q[0];
ry(-0.6038848196977947) q[1];
rz(1.0386543580241208) q[1];
ry(1.715803455732706) q[2];
rz(2.148972422646457) q[2];
ry(0.3581304712572848) q[3];
rz(2.458407671370145) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(3.0322668567876283) q[0];
rz(0.45949373300088864) q[0];
ry(-2.2618155008090257) q[1];
rz(-1.3105282688726885) q[1];
ry(2.2696707134831993) q[2];
rz(-1.347378237606979) q[2];
ry(2.966004823780315) q[3];
rz(-1.291023834282832) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.8170299845182676) q[0];
rz(-1.712578091308817) q[0];
ry(1.8417391673175052) q[1];
rz(-2.13291940689256) q[1];
ry(2.367497906744911) q[2];
rz(-2.831819831124532) q[2];
ry(-0.7155442861097816) q[3];
rz(0.039880031733497585) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.7553636216614223) q[0];
rz(0.5122080823632473) q[0];
ry(1.6749481923300529) q[1];
rz(0.7448425563327871) q[1];
ry(-2.194625719529209) q[2];
rz(-0.47954768288881944) q[2];
ry(3.028888725127614) q[3];
rz(-1.918052382324066) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.9355925619993517) q[0];
rz(-2.964868843028066) q[0];
ry(-1.5211882752669845) q[1];
rz(-1.85405087114831) q[1];
ry(-0.9874976194167432) q[2];
rz(-1.620396107147766) q[2];
ry(2.1599445679436897) q[3];
rz(0.6792982549848094) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.3297029831211695) q[0];
rz(0.9554866822010109) q[0];
ry(2.9047015985125197) q[1];
rz(-1.751106400315543) q[1];
ry(2.576086314779) q[2];
rz(-2.044006229379538) q[2];
ry(-1.0044069638532296) q[3];
rz(-2.814221334496415) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.7425464316756472) q[0];
rz(0.2472126824194056) q[0];
ry(0.41436653775142546) q[1];
rz(0.6099865213372789) q[1];
ry(-0.33953315249738575) q[2];
rz(0.09120112447494756) q[2];
ry(-0.35665163183552906) q[3];
rz(0.7049854489993345) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.0187783171197382) q[0];
rz(-1.7514114089840642) q[0];
ry(2.053920396355843) q[1];
rz(-1.5494832025000103) q[1];
ry(-0.0023296548138426426) q[2];
rz(0.6092762734864671) q[2];
ry(-3.000220903246829) q[3];
rz(1.0851741435995512) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.6439541281803802) q[0];
rz(-0.27845688687796166) q[0];
ry(0.5886097996295712) q[1];
rz(-2.165982353327414) q[1];
ry(0.36654648265507905) q[2];
rz(-1.7581158893085207) q[2];
ry(2.0927668268895774) q[3];
rz(2.1655116433676804) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.24573659138176837) q[0];
rz(0.8253693355870855) q[0];
ry(2.09695059127162) q[1];
rz(0.13262956158402156) q[1];
ry(1.0087645407339654) q[2];
rz(-0.0945396748389907) q[2];
ry(-2.582456428251547) q[3];
rz(-2.976360431539041) q[3];