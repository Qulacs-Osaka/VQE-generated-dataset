OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(2.361320639058698) q[0];
rz(-1.910731199133342) q[0];
ry(2.2948597129865083) q[1];
rz(-1.9646218381111495) q[1];
ry(1.1542561593883391) q[2];
rz(-0.3258369992921183) q[2];
ry(1.374025978914639) q[3];
rz(1.057355174797286) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.33354375470562303) q[0];
rz(1.464529668468832) q[0];
ry(2.895400029790684) q[1];
rz(3.080139614465912) q[1];
ry(-1.1473156402461777) q[2];
rz(0.20634042322017332) q[2];
ry(-0.27208918245712094) q[3];
rz(-0.9280753897122453) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.41268973945904247) q[0];
rz(1.4748437009821405) q[0];
ry(2.1814947545810672) q[1];
rz(-1.4366195871992264) q[1];
ry(-1.086286894568656) q[2];
rz(1.8848778716236485) q[2];
ry(-1.3744384492985076) q[3];
rz(-2.4836598487203574) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.5721974831938338) q[0];
rz(-2.8446821287272703) q[0];
ry(-3.1075256273914498) q[1];
rz(-0.23141686732871045) q[1];
ry(2.8939640912833005) q[2];
rz(-0.33250558418683435) q[2];
ry(0.6133967124457174) q[3];
rz(-2.5513920602163944) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.7761600420123466) q[0];
rz(0.9436233958931443) q[0];
ry(0.544405783902735) q[1];
rz(1.6966841852568653) q[1];
ry(-2.2553776674987374) q[2];
rz(2.1707760184452707) q[2];
ry(-2.0414626915638303) q[3];
rz(-1.3835033375543893) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.22080530022551859) q[0];
rz(3.113732545813646) q[0];
ry(1.6873692153663626) q[1];
rz(-2.957425571023977) q[1];
ry(-2.3590332433497316) q[2];
rz(-3.1214888042019524) q[2];
ry(1.7499740033549938) q[3];
rz(1.139072453710422) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.0059098127750097) q[0];
rz(-2.6951125306858157) q[0];
ry(0.08731663698488819) q[1];
rz(-0.7711743426097937) q[1];
ry(-1.165243708278175) q[2];
rz(1.362534278151126) q[2];
ry(2.30004327747336) q[3];
rz(-0.6399740760450366) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.4349208367554525) q[0];
rz(2.8416357468548505) q[0];
ry(-1.1195733553998632) q[1];
rz(2.5760320106305215) q[1];
ry(-3.119413300438174) q[2];
rz(0.29056723192803485) q[2];
ry(-1.2438263652167416) q[3];
rz(0.9633475117678593) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.0918915460218814) q[0];
rz(2.865760500857361) q[0];
ry(0.04694002263475739) q[1];
rz(-2.667031611982799) q[1];
ry(-2.0425494227642735) q[2];
rz(2.4017575966067533) q[2];
ry(2.9269380347265224) q[3];
rz(-1.986635020405908) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.5949243578895438) q[0];
rz(2.8938304349971284) q[0];
ry(-1.6535467963293558) q[1];
rz(-1.3249665253596565) q[1];
ry(-0.4748788885908723) q[2];
rz(-1.1807809130618487) q[2];
ry(2.998716128868784) q[3];
rz(-2.8107040852803706) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.77791176013049) q[0];
rz(1.6036205575327944) q[0];
ry(1.609422633913444) q[1];
rz(2.3801906654810456) q[1];
ry(0.6971505509212256) q[2];
rz(-2.6787688166562424) q[2];
ry(-1.5433231500715159) q[3];
rz(1.391172939798837) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.5034284756822358) q[0];
rz(-2.2824828914189492) q[0];
ry(1.1616718694101689) q[1];
rz(0.3334194360817164) q[1];
ry(-1.9102473560181963) q[2];
rz(1.0191991201910031) q[2];
ry(0.14661047713359068) q[3];
rz(-1.9610392839773239) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.7742464520528136) q[0];
rz(1.0802801089857066) q[0];
ry(-2.1978151784782893) q[1];
rz(-2.653744598646596) q[1];
ry(-0.456420625354184) q[2];
rz(1.6956844398334463) q[2];
ry(0.49464420723716973) q[3];
rz(-2.096768353921288) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.2368793480238134) q[0];
rz(2.74012274813305) q[0];
ry(0.9008366690829374) q[1];
rz(-1.1784540680514466) q[1];
ry(2.8071990134652385) q[2];
rz(0.4319365447546649) q[2];
ry(-0.4211596078505983) q[3];
rz(2.4624601979538885) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.7452357665315503) q[0];
rz(-1.67330153281865) q[0];
ry(0.5862819687555172) q[1];
rz(0.30704822226122985) q[1];
ry(-2.3190637080360776) q[2];
rz(-0.8328308712903613) q[2];
ry(0.5446979180253182) q[3];
rz(-2.2949991628665534) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.52905625216935) q[0];
rz(0.4047077166052073) q[0];
ry(2.535528007450305) q[1];
rz(2.2255029491645173) q[1];
ry(2.727698537160718) q[2];
rz(-0.08384643210458785) q[2];
ry(2.922526918113889) q[3];
rz(2.0380874819615116) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.3989116231786252) q[0];
rz(-3.007669207193972) q[0];
ry(2.8604578880532188) q[1];
rz(2.218434504220646) q[1];
ry(0.9317355843908657) q[2];
rz(-0.5691952603311264) q[2];
ry(2.0750293533570288) q[3];
rz(0.38371685845613346) q[3];