OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.4639684133545696) q[0];
rz(-2.578329710311237) q[0];
ry(-3.1195182398379644) q[1];
rz(0.056371772448080115) q[1];
ry(2.083798766361117) q[2];
rz(-0.33672453587677703) q[2];
ry(2.6277592667252603) q[3];
rz(2.8189100583638598) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.7580672164425761) q[0];
rz(-0.8451799559260077) q[0];
ry(-0.4861532176411475) q[1];
rz(0.3226453121015851) q[1];
ry(-1.399954269966839) q[2];
rz(1.888769273383395) q[2];
ry(-1.6355614290484581) q[3];
rz(2.144742279015662) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.4790521804332313) q[0];
rz(-2.4907689508261313) q[0];
ry(-0.08754563449241655) q[1];
rz(1.0209531989544676) q[1];
ry(-2.494741570805844) q[2];
rz(2.64982248517) q[2];
ry(0.8927440875848046) q[3];
rz(-1.969496889976843) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.0156559463781942) q[0];
rz(1.8477462221858816) q[0];
ry(3.0947085864303046) q[1];
rz(0.7992406334931844) q[1];
ry(-2.263168590280422) q[2];
rz(-1.1459158577046766) q[2];
ry(-2.189135352614864) q[3];
rz(2.8582977791060684) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.543668972061155) q[0];
rz(-2.883064837802617) q[0];
ry(-3.029593515116783) q[1];
rz(2.9706263811347857) q[1];
ry(-0.419743872414589) q[2];
rz(2.154397297781485) q[2];
ry(3.0878059942022698) q[3];
rz(-1.6646051659443717) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.7989354415481547) q[0];
rz(1.4489287889735212) q[0];
ry(-0.5303113374197976) q[1];
rz(-0.5171907646424749) q[1];
ry(1.5139203956700813) q[2];
rz(-0.5175249737068867) q[2];
ry(-2.085342905972589) q[3];
rz(-2.4243128171226616) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.5373003710897591) q[0];
rz(-2.2214029143992997) q[0];
ry(1.7113287096845555) q[1];
rz(2.512047338792096) q[1];
ry(0.3606544287060771) q[2];
rz(-0.41202589089355085) q[2];
ry(2.4132738705912398) q[3];
rz(3.131667792194645) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.3320255031682855) q[0];
rz(0.6401580148008378) q[0];
ry(-1.9830472088689322) q[1];
rz(2.8490997487571743) q[1];
ry(3.0607204407436606) q[2];
rz(1.5487635989028714) q[2];
ry(1.2973708323190491) q[3];
rz(-0.2271389631955083) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.1304172245450876) q[0];
rz(1.560579117983976) q[0];
ry(-1.3116007363821158) q[1];
rz(-2.748373892348404) q[1];
ry(-2.362725917021374) q[2];
rz(1.6604796662432346) q[2];
ry(-2.6982799048352843) q[3];
rz(-1.8376762230854573) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.3971535748118198) q[0];
rz(0.9876782642782178) q[0];
ry(-0.34751801932246895) q[1];
rz(2.1090948586919067) q[1];
ry(-1.7844568902916134) q[2];
rz(-2.7228632704315063) q[2];
ry(0.879307820787254) q[3];
rz(-0.6502099476034325) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.3377448752485863) q[0];
rz(3.079179920141477) q[0];
ry(0.8614674066237153) q[1];
rz(2.8638160690479864) q[1];
ry(-2.471299161844357) q[2];
rz(1.0201281722333162) q[2];
ry(-0.009627214028454753) q[3];
rz(-2.9096141045649673) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-3.011088585586015) q[0];
rz(-1.879437438037045) q[0];
ry(-2.8208503382920984) q[1];
rz(0.5203610637086857) q[1];
ry(-3.1218315020368297) q[2];
rz(-1.891997283571781) q[2];
ry(2.604331711865593) q[3];
rz(-0.166669700996934) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.398218671628119) q[0];
rz(3.0245754872862403) q[0];
ry(-2.3521011044339426) q[1];
rz(0.6323122485224733) q[1];
ry(-1.226393992352699) q[2];
rz(-0.771900088251151) q[2];
ry(-1.1849684241341156) q[3];
rz(2.361690978615954) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.863150722949792) q[0];
rz(-0.5826944732108803) q[0];
ry(-0.5254266360341011) q[1];
rz(0.8416235862799119) q[1];
ry(2.837606207197319) q[2];
rz(2.388833999384005) q[2];
ry(1.5361660719734445) q[3];
rz(0.5307387284175443) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.5499543493445449) q[0];
rz(1.1749790004997558) q[0];
ry(0.6078981129881988) q[1];
rz(-0.05706049901249077) q[1];
ry(1.7282663862584322) q[2];
rz(2.406375638960534) q[2];
ry(-2.6823057162019666) q[3];
rz(-2.301829107513968) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.8419870167531645) q[0];
rz(-0.22318424974273565) q[0];
ry(-0.0025570707940545377) q[1];
rz(-2.031389551715101) q[1];
ry(1.5688852819717136) q[2];
rz(1.5024824654398212) q[2];
ry(-0.5799737736279118) q[3];
rz(-1.3116258613938552) q[3];